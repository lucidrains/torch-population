from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from fnmatch import fnmatch
from math import prod
from numbers import Integral
from typing import TYPE_CHECKING

import torch
from einops import pack
from torch import Tensor, is_tensor
from torch.nn import Module, Parameter

if TYPE_CHECKING:
    from torch_population.parameters import PopulationParameters

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def is_sequence(v) -> bool:
    return isinstance(v, Sequence) and not isinstance(v, (str, bytes))

def check_pop_size(pop_size: int):
    if isinstance(pop_size, bool) or not isinstance(pop_size, int) or pop_size < 1:
        raise ValueError(f'pop_size must be a positive integer, got {pop_size!r}')

def is_int_index(k) -> bool:
    if isinstance(k, bool):
        return False
    if isinstance(k, Integral):
        return True
    return is_tensor(k) and k.ndim == 0 and not k.is_floating_point() and not k.is_complex() and k.dtype != torch.bool

def has_pop_dim(t: Tensor, pop_size: int | None) -> bool:
    return exists(pop_size) and t.ndim >= 1 and t.shape[0] == pop_size

def is_py_scalar(v) -> bool:
    return isinstance(v, (bool, int, float))

def is_state_val(v) -> bool:
    return is_tensor(v) or is_py_scalar(v)

def to_state_tensor(v) -> Tensor:
    if is_tensor(v):
        return v
    dtype = torch.bool if isinstance(v, bool) else (torch.int64 if isinstance(v, int) else torch.float64)
    return torch.tensor(v, dtype = dtype)

def normalize_index(index, size: int) -> int:
    if not is_int_index(index):
        raise TypeError(f'individual index must be an integer, got {type(index).__name__}')
    idx = int(index.item()) if is_tensor(index) else int(index)
    if idx < 0:
        idx += size
    if not (0 <= idx < size):
        raise IndexError(f'individual index {index} out of range for population of size {size}')
    return idx

def resolve_indices(selector, size: int, *, non_empty: bool = False) -> list[int]:
    if is_int_index(selector):
        indices = [normalize_index(selector, size)]
    elif isinstance(selector, slice):
        indices = list(range(size))[selector]
    elif is_tensor(selector) and selector.dtype == torch.bool:
        if selector.shape != (size,):
            raise ValueError(f'boolean mask must have shape ({size},), got {tuple(selector.shape)}')
        indices = selector.nonzero(as_tuple = False).flatten().tolist()
    elif is_tensor(selector):
        if selector.ndim != 1 or selector.is_floating_point() or selector.is_complex():
            raise IndexError(f'index tensor must be 1D int/bool, got {tuple(selector.shape)}')
        indices = [normalize_index(i, size) for i in selector.tolist()]
    elif is_sequence(selector):
        if selector and all(isinstance(x, bool) for x in selector):
            if len(selector) != size:
                raise ValueError(f'boolean mask must have length {size}, got {len(selector)}')
            indices = [i for i, keep in enumerate(selector) if keep]
        else:
            indices = [normalize_index(i, size) for i in selector]
    else:
        raise TypeError(f'unsupported selector type {type(selector).__name__}')

    if non_empty and not indices:
        raise ValueError('cannot index an empty population')
    return indices

def matches_pattern(name: str, pattern) -> bool:
    if not pattern or pattern is True:
        return bool(pattern)
    if callable(pattern):
        return bool(pattern(name))
    if isinstance(pattern, str):
        return name == pattern or name.startswith(f'{pattern}.') or fnmatch(name, pattern)
    return any(matches_pattern(name, p) for p in pattern)

def init_module(module: Module, init):
    if callable(init):
        init(module)
    elif init == 'reinit':
        for sub in module.modules():
            reset = getattr(sub, 'reset_parameters', None)
            if callable(reset):
                reset()

def clone_template(
    template: Module,
    params: Mapping[str, Tensor] | None = None,
    buffers: Mapping[str, Tensor] | None = None,
    *,
    training: bool | None = None,
) -> Module:
    cloned = deepcopy(template)
    fill = exists(params)

    # remap by id so tied weights and shared buffers stay aliased
    param_map = {
        id(p): Parameter(params[n].detach().clone() if fill else p.detach().to('meta'), requires_grad = p.requires_grad)
        for n, p in cloned.named_parameters()
    }
    buffer_map = {
        id(b): (buffers[n].detach().clone() if fill else b.detach().to('meta'))
        for n, b in cloned.named_buffers()
    }

    for sub in cloned.modules():
        for name, p in sub._parameters.items():
            if exists(p):
                sub._parameters[name] = param_map[id(p)]
        for name, b in sub._buffers.items():
            if exists(b):
                sub._buffers[name] = buffer_map[id(b)]

    if exists(training):
        cloned.train(training)
    return cloned

def slice_pop(t: Tensor, indices: list[int]) -> Tensor:
    return t.detach()[indices]

def pad_pop(t: Tensor, row: Tensor | None = None) -> Tensor:
    pad = row.detach().to(t) if exists(row) else torch.zeros((1, *t.shape[1:]), device = t.device, dtype = t.dtype)
    return torch.cat((t.detach(), pad), dim = 0)

def scatter_pop(base: Tensor, indices: list[int], vals: Sequence[Tensor]) -> Tensor:
    if len(indices) == base.shape[0] and indices == list(range(base.shape[0])):
        return torch.stack(list(vals))
    by_idx = {i: v.to(base.device) for i, v in zip(indices, vals)}
    return torch.stack([by_idx[i] if i in by_idx else base[i] for i in range(base.shape[0])])

def unpack_tensors(flat: Tensor, shapes: list[tuple[int, ...]], sizes: list[int] | None = None) -> list[Tensor]:
    if not shapes:
        return []
    lead = flat.shape[:-1]
    split_sizes = sizes if exists(sizes) else [prod(s) for s in shapes]
    return [t.view(*lead, *shape) for t, shape in zip(flat.split(split_sizes, dim = -1), shapes)]

def slice_spec(spec: TensorSpec, flat: Tensor) -> Tensor:
    return flat[..., spec.offset : spec.offset + spec.numel].view(*flat.shape[:-1], *spec.shape)

def _reduce_mean(t: Tensor) -> Tensor:
    if t.is_floating_point():
        return t.mean(dim = 0)
    if t.dtype == torch.bool:
        return t.any(dim = 0)
    return t.float().mean(dim = 0).round().to(t.dtype)

_REDUCTIONS: dict[str, Callable[[Tensor], Tensor]] = {
    'first': lambda t: t[0],
    'last': lambda t: t[-1],
    'zero': lambda t: torch.zeros_like(t[0]),
    'mean': _reduce_mean,
    'median': lambda t: t.median(dim = 0).values,
    'sum': lambda t: t.sum(dim = 0).to(t.dtype),
    'max': lambda t: t.amax(dim = 0),
    'min': lambda t: t.amin(dim = 0),
    'prod': lambda t: t.prod(dim = 0).to(t.dtype),
}

def reduce_pop(t: Tensor, reduction = 'mean') -> Tensor:
    if callable(reduction):
        return reduction(t)
    if is_int_index(reduction):
        return t[normalize_index(reduction, t.shape[0])]
    if reduction in _REDUCTIONS:
        return _REDUCTIONS[reduction](t)
    raise ValueError(f'unsupported reduction {reduction!r}')

def encode_buffer_key(key: str) -> str:
    return 'k/' + key.replace('%', '%25').replace('.', '%2E')

# records

@dataclass(frozen = True)
class TensorSpec:
    name: str
    shape: tuple[int, ...]
    numel: int
    offset: int
    requires_grad: bool
    trainable_index: int | None = None
    shared: bool = False

@dataclass(frozen = True)
class ParamSlot:
    name: str
    param: Tensor
    grad: Tensor | None
    state: dict[str, Tensor] = field(default_factory = dict)

@dataclass
class SlotSpec:
    shared: bool
    groups: dict[str, list[TensorSpec]]

@dataclass
class AuxMeta:
    shared: bool
    cpu_scalar: bool = False
    py_type: type | None = None

# colocated buffer bank

class TensorStore(Module):
    def __init__(self):
        super().__init__()
        self._populated: dict[str, bool | None] = {}
        self._repeat_on_append: set[str] = set()

    def set(self, key: str, tensor: Tensor, *, populated: bool | None = None, persistent: bool | None = None, append_repeat: bool = False) -> Tensor:
        buf_key = encode_buffer_key(key)
        if not exists(persistent):
            persistent = buf_key not in self._non_persistent_buffers_set
        self._populated[key] = default(populated, self._populated.get(key))
        if append_repeat:
            self._repeat_on_append.add(key)
        self.register_buffer(buf_key, tensor, persistent = persistent)
        return tensor

    def get(self, key: str) -> Tensor | None:
        return self._buffers[encode_buffer_key(key)] if key in self._populated else None

    def __getitem__(self, key: str) -> Tensor:
        if key not in self._populated:
            raise KeyError(key)
        return self._buffers[encode_buffer_key(key)]

    def __contains__(self, key: str) -> bool:
        return key in self._populated

    def __len__(self) -> int:
        return len(self._populated)

    def __iter__(self):
        return iter(self._populated)

    def items(self, prefix: str = '') -> list[tuple[str, Tensor]]:
        cut = len(prefix)
        return [(k[cut:], self._buffers[encode_buffer_key(k)]) for k in self._populated if k.startswith(prefix)]

    def as_dict(self, prefix: str = '') -> dict[str, Tensor]:
        return dict(self.items(prefix))

    def is_populated(self, key: str, pop_size: int | None = None) -> bool:
        if key not in self._populated:
            return False
        flag = self._populated[key]
        return flag if exists(flag) else has_pop_dim(self[key], pop_size)

    @torch.no_grad()
    def upsert_(
        self,
        key: str,
        value: Tensor,
        *,
        decay: float | None = None,
        idx: list[int] | None = None,
        pop_size: int | None = None,
        populated: bool | None = None,
        persistent: bool | None = None,
    ) -> Tensor:
        t = value.detach()
        cur = self.get(key)
        has_pop = has_pop_dim(t, pop_size)

        if not exists(cur):
            if not (exists(idx) and exists(pop_size)):
                return self.set(key, t.clone(), populated = default(populated, has_pop), persistent = persistent)
            batched = has_pop or (t.ndim >= 1 and t.shape[0] == len(idx))
            buf = torch.zeros((pop_size, *(t.shape[1:] if batched else t.shape)), device = t.device, dtype = t.dtype)
            buf[idx] = t[idx] if has_pop else t
            return self.set(key, buf, populated = True, persistent = persistent)

        src = t.to(cur)
        if exists(idx) and self.is_populated(key, pop_size):
            src = src[idx] if has_pop else src
            cur[idx] = cur[idx].lerp(src, 1. - decay) if exists(decay) else src
            return cur
        if exists(decay):
            return cur.lerp_(src, 1. - decay)
        if cur.shape == src.shape:
            return cur.copy_(src)
        return self.set(key, src.clone(), populated = populated, persistent = persistent)

    @torch.no_grad()
    def clear_(
        self,
        prefix: str = '',
        *,
        keys: str | Iterable[str] | None = None,
        pred: Callable[[str], bool] | None = None,
        indices: list[int] | None = None,
        pop_size: int | None = None,
        zero: bool = False,
    ):
        if exists(keys):
            key_set = {f'{prefix}{keys}'} if isinstance(keys, str) else {f'{prefix}{k}' for k in keys}
            matched = [k for k in self._populated if k in key_set]
        else:
            matched = [k for k in self._populated if k.startswith(prefix)]
        if exists(pred):
            matched = [k for k in matched if pred(k)]

        if not exists(indices) and not zero:
            for key in matched:
                self._populated.pop(key, None)
                self._repeat_on_append.discard(key)
                buf_key = encode_buffer_key(key)
                self._buffers.pop(buf_key, None)
                self._non_persistent_buffers_set.discard(buf_key)
            return

        for key in matched:
            if not exists(indices):
                self[key].zero_()
            elif self.is_populated(key, pop_size):
                self[key][indices] = 0

    def remove_(self, prefix: str = '', *, pred: Callable[[str], bool] | None = None):
        self.clear_(prefix, pred = pred)

    def zero_(self, prefix: str = '', *, pred: Callable[[str], bool] | None = None, indices: list[int] | None = None, pop_size: int | None = None):
        self.clear_(prefix, pred = pred, indices = indices, pop_size = pop_size, zero = True)

    @torch.no_grad()
    def copy_from_(self, other: TensorStore, *, prefixes: tuple[str, ...] | None = None, indices: list[int] | None = None, pop_size: int | None = None):
        for key in self._populated:
            if exists(prefixes) and not key.startswith(prefixes):
                continue
            src = other.get(key)
            if not exists(src):
                continue
            dst = self[key]
            if not exists(indices):
                dst.copy_(src.detach().to(dst))
            elif self.is_populated(key, pop_size):
                dst[indices] = src[indices].detach().to(dst)

    @torch.no_grad()
    def keep_(self, indices: list[int], pop_size: int | None = None):
        for key in self._populated:
            if self.is_populated(key, pop_size):
                buf_key = encode_buffer_key(key)
                self._buffers[buf_key] = slice_pop(self._buffers[buf_key], indices)

    @torch.no_grad()
    def append_(self, pop_size: int | None = None):
        for key in self._populated:
            if self.is_populated(key, pop_size):
                buf_key = encode_buffer_key(key)
                cur = self._buffers[buf_key]
                self._buffers[buf_key] = pad_pop(cur, cur[:1] if key in self._repeat_on_append else None)

    def clone(self, *, detach: bool = True, prefixes: tuple[str, ...] | None = None, shallow_prefixes: tuple[str, ...] = ()) -> TensorStore:
        out = TensorStore()
        for key, populated in self._populated.items():
            if exists(prefixes) and not key.startswith(prefixes):
                continue
            buf_key = encode_buffer_key(key)
            tensor = self._buffers[buf_key]
            if not (shallow_prefixes and key.startswith(shallow_prefixes)):
                tensor = tensor.detach().clone() if detach else tensor.clone()
            out.set(
                key, tensor,
                populated = populated,
                persistent = buf_key not in self._non_persistent_buffers_set,
                append_repeat = key in self._repeat_on_append,
            )
        return out

# prefix handle into PopulationParameters.store for optimizer state

class OptStore:
    def __init__(self, pp: PopulationParameters, name: str):
        self._pp = pp
        self.name = name
        self._prefix = f'opt/{name}/'
        self.slots: dict[str, SlotSpec] = {}
        self.aux: dict[str, dict[str, AuxMeta]] = {}

    def set(self, key: str, tensor: Tensor, *, populated: bool | None = None, append_repeat: bool = False) -> Tensor:
        return self._pp.store.set(self._prefix + key, tensor, populated = populated, append_repeat = append_repeat)

    def put(self, key: str, value: Tensor, *, populated: bool, inplace: bool):
        if not inplace:
            self.set(key, value, populated = populated)
        elif value is not self[key]:
            self[key].copy_(value)

    def get(self, key: str) -> Tensor | None:
        return self._pp.store.get(self._prefix + key)

    def __getitem__(self, key: str) -> Tensor:
        return self._pp.store[self._prefix + key]

    def __contains__(self, key: str) -> bool:
        return (self._prefix + key) in self._pp.store

    def items(self, subprefix: str = '') -> list[tuple[str, Tensor]]:
        return self._pp.store.items(self._prefix + subprefix)

    def is_populated(self, key: str, pop_size: int | None = None) -> bool:
        return self._pp.store.is_populated(self._prefix + key, default(pop_size, self._pp.pop_size))

    def clear(self, *, keep_hp: bool = False):
        hp_prefix = f'{self._prefix}hp/'
        pred = (lambda k: not k.startswith(hp_prefix)) if keep_hp else None
        self._pp.store.remove_(self._prefix, pred = pred)
        self.slots.clear()
        self.aux.clear()

    def alloc_slot(
        self,
        slot: str,
        param_names: Sequence[str] | None = None,
        *,
        shared: bool = False,
        dtypes: Mapping[str, torch.dtype] | None = None,
    ) -> OptStore:
        if slot in self.slots:
            return self

        pp = self._pp
        named = pp.named_parameters_dict()
        allowed = set(param_names) if exists(param_names) else None

        selected = [s for s in pp.trainable_specs if not exists(allowed) or s.name in allowed]
        pop_specs = [s for s in selected if not s.shared]
        shared_specs = [s for s in selected if s.shared]
        lead = None if (shared or not pp.is_population) else pp.pop_size
        groups: dict[str, list[TensorSpec]] = {}

        for suffix, specs, pop_dim in (('pop', pop_specs, lead), ('sh', shared_specs, None)):
            if not specs:
                continue
            first = named[specs[0].name]
            dtype = dtypes.get(specs[0].name, first.dtype) if exists(dtypes) else first.dtype
            numel = sum(s.numel for s in specs)
            shape = (pop_dim, numel) if exists(pop_dim) else (numel,)
            self.set(f'slot/{slot}/{suffix}', torch.zeros(shape, device = first.device, dtype = dtype), populated = exists(pop_dim))
            groups[suffix] = specs

        self.slots[slot] = SlotSpec(shared = shared, groups = groups)
        return self

    def alloc_aux(
        self,
        key: str,
        param: str,
        shape: tuple[int, ...],
        *,
        dtype: torch.dtype,
        device: torch.device,
        shared: bool = False,
        cpu_scalar: bool | None = None,
        py_type: type | None = None,
    ) -> Tensor:
        buf_key = f'aux/{key}/{param}'
        params = self.aux.setdefault(key, {})
        if param in params:
            return self[buf_key]

        pp = self._pp
        is_shared = shared or pp.is_shared_param(param)
        is_pop = pp.is_population and not is_shared
        is_cpu = default(cpu_scalar, exists(py_type) or (shape == () and device.type == 'cpu'))
        params[param] = AuxMeta(shared = is_shared, cpu_scalar = is_cpu, py_type = py_type)

        full_shape = (pp.pop_size, *shape) if is_pop else shape
        return self.set(buf_key, torch.zeros(full_shape, device = device, dtype = dtype), populated = is_pop)

    def state_keys(self) -> list[str]:
        return list(dict.fromkeys((*self.slots, *self.aux)))

    def aux_keys_for(self, param: str) -> list[str]:
        return [key for key, params in self.aux.items() if param in params]

    def slot_views(self, slot: str) -> dict[str, Tensor]:
        views: dict[str, Tensor] = {}
        for suffix, specs in self.slots[slot].groups.items():
            buf = self[f'slot/{slot}/{suffix}']
            views.update(zip((s.name for s in specs), unpack_tensors(buf, [s.shape for s in specs])))
        return views

    def all_views(self) -> dict[str, dict[str, Tensor]]:
        return {slot: self.slot_views(slot) for slot in self.slots}

    def pack_slots_(self, views: dict[str, dict[str, Tensor]], *, is_pop_group: bool):
        suffix = 'pop' if is_pop_group else 'sh'
        for key, slot in self.slots.items():
            specs = slot.groups.get(suffix)
            if not specs:
                continue
            slot_key = f'slot/{key}/{suffix}'
            is_pop = self[slot_key].ndim == 2
            packed, _ = pack([views[key][s.name] for s in specs], 'p *' if is_pop else '*')
            self.set(slot_key, packed, populated = is_pop)

    def alloc_global(self, name: str, init_tensor: Tensor, *, populated: bool = False) -> Tensor:
        key = f'global/{name}'
        if key not in self:
            self.set(key, init_tensor.detach().clone(), populated = populated)
        return self[key]

    def global_tensor(self, name: str) -> Tensor | None:
        return self.get(f'global/{name}')

    def globals_dict(self) -> dict[str, Tensor]:
        return self._pp.store.as_dict(f'{self._prefix}global/')

    def is_global_populated(self, name: str) -> bool:
        return self.is_populated(f'global/{name}')

    def is_init(self, spec: TensorSpec, idx: int | None = None) -> bool:
        flags = self.global_tensor('_init')
        if not exists(flags):
            return True
        col = flags[:, spec.trainable_index]
        return bool(col.any() if (spec.shared or not exists(idx)) else col[idx])

    def mark_init_(self, specs: Sequence[TensorSpec], members: list[int] | None = None, value: bool = True):
        flags = self['global/_init']
        cols = [s.trainable_index for s in specs]
        if not exists(members):
            flags[:, cols] = value
        else:
            for m in members:
                flags[m, cols] = value

    def _has_shared_member_state(self) -> bool:
        return any(slot.shared for slot in self.slots.values()) or any(
            meta.shared
            for params in self.aux.values()
            for pname, meta in params.items()
            if not self._pp.is_shared_param(pname)
        )

    def reset_members_(self, indices: list[int] | None):
        hp_prefix = f'{self._prefix}hp/'
        init_key = f'{self._prefix}global/_init'
        self._pp.store.zero_(
            self._prefix,
            pred = lambda k: not k.startswith(hp_prefix) and k != init_key,
            indices = indices,
        )
        flags = self.global_tensor('_init')
        if not exists(flags):
            return
        if not exists(indices):
            flags.zero_()
        elif not self._has_shared_member_state():
            self.mark_init_([s for s in self._pp.trainable_specs if not s.shared], indices, value = False)

    def clone_handle(self, new_pp: PopulationParameters) -> OptStore:
        out = OptStore(new_pp, self.name)
        out.slots = {k: SlotSpec(s.shared, dict(s.groups)) for k, s in self.slots.items()}
        out.aux = {k: {p: AuxMeta(m.shared, m.cpu_scalar, m.py_type) for p, m in ps.items()} for k, ps in self.aux.items()}
        return out
