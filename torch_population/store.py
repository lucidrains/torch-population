from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from fnmatch import fnmatch
from typing import TYPE_CHECKING

import torch
from einops import unpack
from torch import Tensor, is_tensor
from torch.nn import Module, Parameter

if TYPE_CHECKING:
    from torch_population.parameters import PopulationParameters

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def matches_pattern(name: str, pattern) -> bool:
    if pattern is None or pattern is False:
        return False
    if pattern is True:
        return True
    if callable(pattern):
        return bool(pattern(name))
    if isinstance(pattern, str):
        return pattern in name or fnmatch(name, pattern)
    return any(matches_pattern(name, p) for p in pattern)

def is_int_index(key) -> bool:
    if isinstance(key, bool):
        return False
    if isinstance(key, int):
        return True
    return (
        is_tensor(key)
        and key.ndim == 0
        and not key.is_floating_point()
        and key.dtype != torch.bool
    )

def reset_module_parameters(module: Module):
    for submod in module.modules():
        reset_fn = getattr(submod, 'reset_parameters', None)
        if callable(reset_fn):
            reset_fn()

def named_buffer_persistence(module: Module) -> dict[str, bool]:
    persistence = {}
    seen = set()

    for prefix, submod in module.named_modules():
        non_persistent = submod._non_persistent_buffers_set
        for name, buf in submod._buffers.items():
            if not exists(buf) or id(buf) in seen:
                continue
            seen.add(id(buf))
            full_name = f'{prefix}.{name}' if prefix else name
            persistence[full_name] = name not in non_persistent

    return persistence

def materialize_from_template(
    template: Module,
    params: Mapping[str, Tensor],
    buffers: Mapping[str, Tensor],
    training: bool,
) -> Module:
    out = deepcopy(template)

    param_map = {
        id(p): Parameter(params[n].detach().clone(), requires_grad = params[n].requires_grad)
        for n, p in out.named_parameters(remove_duplicate = True)
    }
    buf_map = {
        id(b): buffers[n].detach().clone()
        for n, b in out.named_buffers(remove_duplicate = True)
    }

    for submod in out.modules():
        for n, p in list(submod._parameters.items()):
            if exists(p):
                submod._parameters[n] = param_map[id(p)]
        for n, b in list(submod._buffers.items()):
            if exists(b):
                submod._buffers[n] = buf_map[id(b)]

    out.train(training)
    return out

def slice_pop(t: Tensor, indices: list[int]) -> Tensor:
    idx = torch.tensor(indices, dtype = torch.long, device = t.device)
    return t.detach()[idx].clone()

def pad_pop(t: Tensor, row: Tensor | None = None) -> Tensor:
    if exists(row):
        pad = row.detach().to(t)
    else:
        pad = torch.zeros((1, *t.shape[1:]), device = t.device, dtype = t.dtype)
    return torch.cat((t.detach(), pad), dim = 0)

def unpack_tensors(
    flat: Tensor,
    shapes: list[tuple[int, ...]],
    idx: int | None = None,
) -> list[Tensor]:
    if not shapes:
        return []
    if flat.ndim == 2:
        if exists(idx):
            return unpack(flat[idx], shapes, '*')
        return unpack(flat, shapes, 'p *')
    return unpack(flat, shapes, '*')

def slice_spec(
    spec: TensorSpec,
    flat: Tensor,
    idx: int | None = None,
    pop_size: int | None = None,
) -> Tensor:
    lo = spec.offset
    hi = spec.offset + spec.numel
    if flat.ndim == 2:
        if exists(idx):
            return flat[idx, lo:hi].view(*spec.shape)
        return flat[:, lo:hi].view(pop_size, *spec.shape)
    return flat[lo:hi].view(*spec.shape)

# specs & slot views

@dataclass(frozen = True)
class TensorSpec:
    name: str
    shape: tuple[int, ...]
    numel: int
    bucket: int
    offset: int
    requires_grad: bool
    trainable_index: int | None = None
    shared: bool = False

@dataclass(frozen = True)
class ParamSlot:
    name: str
    param: Tensor
    grad: Tensor | None
    exp_avg: Tensor | None = None
    exp_avg_sq: Tensor | None = None
    step: Tensor | None = None
    state: dict[str, Tensor] = field(default_factory = dict)

@dataclass
class _BufMeta:
    buf: str
    populated: bool | None = None
    source: str = ''
    persistent: bool = True
    append_repeat: bool = False

@dataclass(frozen = True)
class _SlotBucket:
    key: str
    names: list[str]
    shapes: list[tuple[int, ...]]

# universal colocated buffer store (child Module -> .to(), .half(), state_dict() all automatic)

class TensorStore(Module):
    """Named colocated tensor store with lockstep population resize, reset, and EMA updates."""

    def __init__(self):
        super().__init__()
        self._meta: dict[str, _BufMeta] = {}
        self._next_id: int = 0

    def set(
        self,
        key: str,
        tensor: Tensor,
        *,
        populated: bool | None = None,
        source: str = '',
        persistent: bool = True,
        append_repeat: bool = False,
    ) -> Tensor:
        meta = self._meta.get(key)

        if exists(meta):
            self._buffers[meta.buf] = tensor
            if exists(populated):
                meta.populated = populated
            if source:
                meta.source = source
            meta.persistent = persistent
            if append_repeat:
                meta.append_repeat = True
            return tensor

        buf_name = f'b{self._next_id}'
        self._next_id += 1
        self._meta[key] = _BufMeta(buf_name, populated, source, persistent, append_repeat)
        self.register_buffer(buf_name, tensor, persistent = persistent)
        return tensor

    def get(self, key: str) -> Tensor | None:
        meta = self._meta.get(key)
        return self._buffers[meta.buf] if exists(meta) else None

    def __getitem__(self, key: str) -> Tensor:
        t = self.get(key)
        if not exists(t):
            raise KeyError(key)
        return t

    def __contains__(self, key: str) -> bool:
        return key in self._meta

    def __len__(self) -> int:
        return len(self._meta)

    def __iter__(self):
        return iter(self._meta)

    def keys(self) -> list[str]:
        return list(self._meta)

    def items(self) -> list[tuple[str, Tensor]]:
        return [(k, self._buffers[m.buf]) for k, m in self._meta.items()]

    def as_dict(self) -> dict[str, Tensor]:
        return {k: self._buffers[m.buf] for k, m in self._meta.items()}

    def source_of(self, key: str) -> str:
        meta = self._meta.get(key)
        return meta.source if exists(meta) else ''

    def is_populated(self, key: str, pop_size: int | None) -> bool:
        meta = self._meta.get(key)
        if not exists(pop_size) or not exists(meta):
            return False
        if exists(meta.populated):
            return meta.populated
        buf = self._buffers[meta.buf]
        return buf.ndim >= 1 and buf.shape[0] == pop_size

    def upsert_(
        self,
        key: str,
        value: Tensor,
        *,
        decay: float | None = None,
        idx: list[int] | None = None,
        pop_size: int | None = None,
        source: str = '',
        populated: bool | None = None,
    ) -> Tensor:
        val = value.detach()
        existing = self.get(key)

        if not exists(existing):
            if exists(idx) and exists(pop_size):
                init_t = torch.zeros((pop_size, *val.shape[1:]), device = val.device, dtype = val.dtype)
                src = val[idx] if (val.ndim >= 1 and val.shape[0] == pop_size) else val
                init_t[idx] = src
                pop_flag = True
            else:
                init_t = val.clone()
                auto_pop = exists(pop_size) and init_t.ndim >= 1 and init_t.shape[0] == pop_size
                pop_flag = populated if exists(populated) else auto_pop
            return self.set(key, init_t, populated = pop_flag, source = source)

        with torch.no_grad():
            v = val.to(existing)
            is_pop_slice = exists(idx) and exists(pop_size) and existing.ndim >= 1 and existing.shape[0] == pop_size

            if is_pop_slice:
                src = v[idx] if (v.ndim >= 1 and v.shape[0] == pop_size) else v
                existing[idx] = existing[idx].lerp(src, 1. - decay) if exists(decay) else src
            elif exists(decay):
                existing.lerp_(v, 1. - decay)
            elif existing.shape == v.shape:
                existing.copy_(v)
            else:
                return self.set(key, v.clone(), populated = populated, source = source)

        return existing

    def remove(self, key: str):
        meta = self._meta.pop(key, None)
        if not exists(meta):
            return
        self._buffers.pop(meta.buf, None)
        self._non_persistent_buffers_set.discard(meta.buf)

    def clear(self):
        self._buffers.clear()
        self._non_persistent_buffers_set.clear()
        self._meta.clear()

    def reset_(
        self,
        indices: list[int] | None = None,
        *,
        pop_size: int | None = None,
        pred: Callable[[str], bool] | None = None,
    ):
        with torch.no_grad():
            for k in list(self._meta):
                if exists(pred) and not pred(k):
                    continue
                if not exists(indices):
                    self[k].zero_()
                elif self.is_populated(k, pop_size):
                    self[k][indices] = 0

    def clear_keys_(
        self,
        selector = None,
        *,
        indices: list[int] | None = None,
        pop_size: int | None = None,
    ):
        if not exists(selector):
            pred = None
        elif callable(selector):
            pred = selector
        elif isinstance(selector, str):
            pred = lambda k: k == selector
        else:
            allowed = set(selector)
            pred = lambda k: k in allowed

        if exists(indices):
            self.reset_(indices, pop_size = pop_size, pred = pred)
        elif not exists(pred):
            self.clear()
        else:
            for k in list(self._meta):
                if pred(k):
                    self.remove(k)

    def copy_from_(
        self,
        other: TensorStore,
        *,
        indices: list[int] | None = None,
        pop_size: int | None = None,
    ):
        with torch.no_grad():
            for k in self._meta:
                src = other.get(k)
                if not exists(src):
                    continue
                dst = self[k]
                val = src.detach().to(dst)
                if not exists(indices):
                    dst.copy_(val)
                elif self.is_populated(k, pop_size):
                    dst[indices] = val[indices]

    def keep_(self, indices: list[int], old_pop: int):
        with torch.no_grad():
            for k, meta in self._meta.items():
                if self.is_populated(k, old_pop):
                    self._buffers[meta.buf] = slice_pop(self._buffers[meta.buf], indices)

    def append_(self, old_pop: int):
        with torch.no_grad():
            for k, meta in self._meta.items():
                if self.is_populated(k, old_pop):
                    buf = self._buffers[meta.buf]
                    row = buf[:1] if meta.append_repeat else None
                    self._buffers[meta.buf] = pad_pop(buf, row)

    def _copy_buffers_into(self, out: TensorStore) -> TensorStore:
        for k, meta in self._meta.items():
            out.set(
                k,
                self._buffers[meta.buf].detach().clone(),
                populated = meta.populated,
                source = meta.source,
                persistent = meta.persistent,
                append_repeat = meta.append_repeat,
            )
        return out

    def clone_store(self) -> TensorStore:
        return self._copy_buffers_into(TensorStore())


class StoreDict(Module):
    """Module mapping arbitrary string keys (including dotted names) to colocated child stores."""

    def __init__(self):
        super().__init__()
        self._key_to_mod: dict[str, str] = {}
        self._next_id: int = 0

    def __setitem__(self, key: str, module: Module):
        mod_name = self._key_to_mod.get(key)
        if not exists(mod_name):
            mod_name = f's{self._next_id}'
            self._next_id += 1
            self._key_to_mod[key] = mod_name
        self.add_module(mod_name, module)

    def __getitem__(self, key: str):
        if key not in self._key_to_mod:
            raise KeyError(key)
        return self._modules[self._key_to_mod[key]]

    def __contains__(self, key: str) -> bool:
        return key in self._key_to_mod

    def __len__(self) -> int:
        return len(self._key_to_mod)

    def get(self, key: str, fallback = None):
        return self[key] if key in self._key_to_mod else fallback

    def keys(self) -> list[str]:
        return list(self._key_to_mod)

    def values(self) -> list:
        return [self._modules[m] for m in self._key_to_mod.values()]

    def items(self) -> list[tuple[str, Module]]:
        return [(k, self._modules[m]) for k, m in self._key_to_mod.items()]

# generic optimizer store (backs Adam, AdamW, SGD, Muon, and custom optimizers)

class OptStore(TensorStore):
    """Unified colocated optimizer store for any optimizer (Adam, AdamW, SGD, Muon, custom)."""

    def __init__(self, pp: PopulationParameters):
        super().__init__()
        self._pp_ref: list[PopulationParameters] = [pp]
        self._slots: dict[str, dict] = {}
        self._globals: list[str] = []
        self._hps: list[str] = []
        self._opt_meta: dict = {}

    @property
    def _pp(self) -> PopulationParameters:
        return self._pp_ref[0]

    def alloc_slot(
        self,
        slot: str,
        param_names: Sequence[str] | None = None,
        *,
        shared: bool = False,
    ) -> OptStore:
        if slot in self._slots:
            return self

        pp = self._pp
        allowed = set(param_names) if exists(param_names) else None
        named_p = pp.named_parameters_dict()
        selected = [s for s in pp.trainable_specs if not exists(allowed) or s.name in allowed]

        def _pack_buckets(specs: list[TensorSpec], prefix: str, lead_pop: int | None) -> list[_SlotBucket]:
            grouped: dict[tuple[torch.device, torch.dtype], list[TensorSpec]] = {}
            for s in specs:
                dev_dt = (named_p[s.name].device, named_p[s.name].dtype)
                grouped.setdefault(dev_dt, []).append(s)

            buckets = []
            for b, ((dev, dt), grp) in enumerate(grouped.items()):
                key = f'{prefix}_{b}'
                total = sum(s.numel for s in grp)
                shape = (lead_pop, total) if exists(lead_pop) else (total,)
                buf = torch.zeros(shape, device = dev, dtype = dt)
                self.set(key, buf, populated = exists(lead_pop))
                buckets.append(_SlotBucket(key, [s.name for s in grp], [s.shape for s in grp]))
            return buckets

        lead_pop = None if (shared or not pp.is_population) else pp.pop_size
        pop_specs = [s for s in selected if not s.shared]
        sh_specs = [s for s in selected if s.shared]

        self._slots[slot] = {
            'shared': shared,
            'pop': _pack_buckets(pop_specs, f'slot/{slot}/pop', lead_pop),
            'sh': _pack_buckets(sh_specs, f'slot/{slot}/sh', None),
        }
        return self

    def reset_slots_and_globals_(self):
        self.clear_keys_(lambda k: k.startswith(('slot/', 'global/')))
        self._slots.clear()
        self._globals.clear()

    def has_slot(self, slot: str) -> bool:
        return slot in self._slots

    def slot_names(self) -> list[str]:
        return list(self._slots)

    def slot_is_shared(self, slot: str) -> bool:
        return bool(self._slots[slot]['shared'])

    def slot_flats(self, slot: str, *, shared_params: bool = False) -> list[Tensor]:
        kind = 'sh' if shared_params else 'pop'
        return [self[b.key] for b in self._slots[slot][kind]]

    def set_slot_flat(self, slot: str, bucket: int, tensor: Tensor, *, shared_params: bool = False):
        info = self._slots[slot]
        kind = 'sh' if shared_params else 'pop'
        self.set(info[kind][bucket].key, tensor, populated = tensor.ndim == 2)
        if not shared_params:
            info['shared'] = tensor.ndim == 1

    def slot_views(self, slot: str, *, idx: int | None = None) -> dict[str, Tensor]:
        info = self._slots[slot]
        out: dict[str, Tensor] = {}
        for b in info['pop']:
            buf = self[b.key]
            use_idx = idx if buf.ndim == 2 else None
            out.update(zip(b.names, unpack_tensors(buf, b.shapes, idx = use_idx)))
        for b in info['sh']:
            out.update(zip(b.names, unpack_tensors(self[b.key], b.shapes)))
        return out

    def alloc_global(self, name: str, init_tensor: Tensor, *, populated: bool = False) -> Tensor:
        key = f'global/{name}'
        if name not in self._globals:
            self._globals.append(name)
            self.set(key, init_tensor.detach().clone(), populated = populated)
        return self[key]

    def set_global(self, name: str, tensor: Tensor, *, populated: bool | None = None):
        if name not in self._globals:
            self._globals.append(name)
        pop_flag = populated if exists(populated) else (tensor.ndim >= 2)
        self.set(f'global/{name}', tensor, populated = pop_flag)

    def has_global(self, name: str) -> bool:
        return name in self._globals

    def is_global_populated(self, name: str) -> bool:
        return self.is_populated(f'global/{name}', self._pp.pop_size)

    def global_tensor(self, name: str) -> Tensor:
        if name not in self._globals:
            raise KeyError(f'no global buffer named {name!r}')
        return self[f'global/{name}']

    def globals_dict(self) -> dict[str, Tensor]:
        return {n: self[f'global/{n}'] for n in self._globals}

    def set_hp(self, name: str, tensor: Tensor):
        if name not in self._hps:
            self._hps.append(name)
        self.set(f'hp/{name}', tensor, populated = tensor.ndim == 1, append_repeat = True)

    def hp_buffer(self, name: str) -> Tensor | None:
        return self.get(f'hp/{name}')

    def hp_buffers_dict(self) -> dict[str, Tensor]:
        return {n: self[f'hp/{n}'] for n in self._hps if f'hp/{n}' in self}

    def reset_members_(self, indices: list[int] | None):
        self.reset_(indices, pop_size = self._pp.pop_size, pred = lambda k: not k.startswith('hp/'))

    def copy_from_functional_(
        self,
        fstate: FunctionalOptState,
        *,
        indices: list[int] | None = None,
        pop_size: int | None = None,
    ):
        def _cp(dst: Tensor, src: Tensor, allow_slice: bool):
            val = src.detach().to(dst)
            if exists(indices) and allow_slice and exists(pop_size) and dst.ndim >= 1 and dst.shape[0] == pop_size:
                dst[indices] = val[indices]
            elif not exists(indices):
                dst.copy_(val)

        with torch.no_grad():
            for slot, src_flats in fstate.slots.items():
                if not self.has_slot(slot):
                    continue
                for d, s in zip(self.slot_flats(slot), src_flats):
                    _cp(d, s, allow_slice = (d.ndim == 2))
            for slot, src_flats in fstate.shared_slots.items():
                if not self.has_slot(slot):
                    continue
                for d, s in zip(self.slot_flats(slot, shared_params = True), src_flats):
                    _cp(d, s, allow_slice = False)
            for gname, src_g in fstate.globals.items():
                if self.has_global(gname):
                    dst_g = self.global_tensor(gname)
                    _cp(dst_g, src_g, allow_slice = self.is_global_populated(gname))

    def clone_for(self, new_pp: PopulationParameters) -> OptStore:
        out = OptStore(new_pp)
        out._slots = deepcopy(self._slots)
        out._globals = list(self._globals)
        out._hps = list(self._hps)
        out._opt_meta = dict(self._opt_meta)
        return self._copy_buffers_into(out)


# generic functional optimizer state for differentiable inner unrolls (DiscoRL / MAML)

@dataclass
class FunctionalOptState:
    slots: dict[str, list[Tensor]] = field(default_factory = dict)
    shared_slots: dict[str, list[Tensor]] = field(default_factory = dict)
    globals: dict[str, Tensor] = field(default_factory = dict)
    hp_buffers: dict[str, Tensor] = field(default_factory = dict)
    meta: dict = field(default_factory = dict)

    def map_tensors(self, fn: Callable[[Tensor], Tensor]) -> FunctionalOptState:
        def _map_meta(v):
            if is_tensor(v):
                return fn(v)
            if isinstance(v, tuple):
                return tuple(fn(x) if is_tensor(x) else x for x in v)
            return v

        return FunctionalOptState(
            slots = {k: [fn(t) for t in vs] for k, vs in self.slots.items()},
            shared_slots = {k: [fn(t) for t in vs] for k, vs in self.shared_slots.items()},
            globals = {k: fn(t) for k, t in self.globals.items()},
            hp_buffers = {k: fn(t) for k, t in self.hp_buffers.items()},
            meta = {k: _map_meta(v) for k, v in self.meta.items()},
        )

    def reset_members_(self, indices: list[int]):
        with torch.no_grad():
            for vs in self.slots.values():
                for t in vs:
                    if t.ndim == 2:
                        t[indices] = 0
            for t in self.globals.values():
                if t.ndim >= 2:
                    t[indices] = 0
