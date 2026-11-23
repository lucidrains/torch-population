from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
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

def is_sequence(v) -> bool:
    return isinstance(v, Sequence) and not isinstance(v, (str, bytes))

def is_int_index(k) -> bool:
    if isinstance(k, bool):
        return False
    if isinstance(k, int):
        return True
    return is_tensor(k) and k.ndim == 0 and not k.is_floating_point() and k.dtype != torch.bool

def has_pop_dim(t: Tensor, pop_size: int | None) -> bool:
    return exists(pop_size) and t.ndim >= 1 and t.shape[0] == pop_size

def normalize_index(index, size: int) -> int:
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
        if selector.ndim != 1 or selector.is_floating_point():
            raise IndexError(f'index tensor must be 1D int/bool, got {tuple(selector.shape)}')
        indices = [normalize_index(i, size) for i in selector.tolist()]
    elif is_sequence(selector):
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
        return pattern in name or fnmatch(name, pattern)
    return any(matches_pattern(name, p) for p in pattern)

def init_module(module: Module, init):
    if callable(init):
        init(module)
    elif init == 'reinit':
        for sub in module.modules():
            reset = getattr(sub, 'reset_parameters', None)
            if callable(reset):
                reset()

def clone_template(template: Module, params: Mapping[str, Tensor] | None = None, buffers: Mapping[str, Tensor] | None = None, *, training: bool | None = None) -> Module:
    cloned = deepcopy(template)
    fill = exists(params)

    # remap by id so tied weights remain aliased
    param_map = {
        id(p): Parameter(params[n].detach().clone() if fill else p.detach().to('meta'), requires_grad = (params[n] if fill else p).requires_grad)
        for n, p in cloned.named_parameters()
    }
    buffer_map = {
        id(b): (buffers[n].detach().clone() if fill else b.detach().to('meta'))
        for n, b in cloned.named_buffers()
    }

    for sub in cloned.modules():
        for n, p in list(sub._parameters.items()):
            if exists(p):
                sub._parameters[n] = param_map[id(p)]
        for n, b in list(sub._buffers.items()):
            if exists(b):
                sub._buffers[n] = buffer_map[id(b)]

    if exists(training):
        cloned.train(training)
    return cloned

def slice_pop(t: Tensor, indices: list[int]) -> Tensor:
    return t.detach()[torch.tensor(indices, dtype = torch.long, device = t.device)].clone()

def pad_pop(t: Tensor, row: Tensor | None = None) -> Tensor:
    pad = row.detach().to(t) if exists(row) else torch.zeros((1, *t.shape[1:]), device = t.device, dtype = t.dtype)
    return torch.cat((t.detach(), pad), dim = 0)

def unpack_tensors(flat: Tensor, shapes: list[tuple[int, ...]], idx: int | None = None) -> list[Tensor]:
    if not shapes:
        return []
    if flat.ndim == 2 and exists(idx):
        return unpack(flat[idx], shapes, '*')
    return unpack(flat, shapes, 'p *' if flat.ndim == 2 else '*')

def slice_spec(spec: TensorSpec, flat: Tensor) -> Tensor:
    sl = flat[..., spec.offset : spec.offset + spec.numel]
    return sl.view(flat.shape[0], *spec.shape) if flat.ndim == 2 else sl.view(*spec.shape)

def reduce_mean(stacked: Tensor, dtype: torch.dtype) -> Tensor:
    if stacked.is_floating_point():
        return stacked.mean(dim = 0).to(dtype)
    if dtype == torch.bool:
        return stacked.any(dim = 0)
    return stacked.float().mean(dim = 0).round().to(dtype)

def _buf_key(key: str) -> str:
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
    pop_names: list[str]
    pop_shapes: list[tuple[int, ...]]
    sh_names: list[str]
    sh_shapes: list[tuple[int, ...]]

@dataclass
class AuxMeta:
    shared: bool
    cpu_scalar: bool = False

# colocated buffer bank

class TensorStore(Module):
    def __init__(self):
        super().__init__()
        self._populated: dict[str, bool | None] = {}
        self._repeat: set[str] = set()

    def set(self, key: str, tensor: Tensor, *, populated: bool | None = None, persistent: bool | None = None, append_repeat: bool = False) -> Tensor:
        buf = _buf_key(key)
        if not exists(persistent):
            persistent = buf not in self._non_persistent_buffers_set
        self._populated[key] = default(populated, self._populated.get(key))
        if append_repeat:
            self._repeat.add(key)
        self.register_buffer(buf, tensor, persistent = persistent)
        return tensor

    def get(self, key: str) -> Tensor | None:
        return self._buffers[_buf_key(key)] if key in self._populated else None

    def __getitem__(self, key: str) -> Tensor:
        if key not in self._populated:
            raise KeyError(key)
        return self._buffers[_buf_key(key)]

    def __contains__(self, key: str) -> bool:
        return key in self._populated

    def __len__(self) -> int:
        return len(self._populated)

    def __iter__(self):
        return iter(self._populated)

    def items(self, prefix: str = '') -> list[tuple[str, Tensor]]:
        cut = len(prefix)
        return [(k[cut:], self._buffers[_buf_key(k)]) for k in self._populated if k.startswith(prefix)]

    def as_dict(self, prefix: str = '') -> dict[str, Tensor]:
        return dict(self.items(prefix))

    def is_populated(self, key: str, pop_size: int | None) -> bool:
        if not exists(pop_size) or key not in self._populated:
            return False
        flag = self._populated[key]
        return flag if exists(flag) else has_pop_dim(self._buffers[_buf_key(key)], pop_size)

    @torch.no_grad()
    def upsert_(self, key: str, value: Tensor, *, decay: float | None = None, idx: list[int] | None = None, pop_size: int | None = None, populated: bool | None = None, persistent: bool | None = None) -> Tensor:
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
        if exists(idx) and has_pop_dim(cur, pop_size):
            src = src[idx] if has_pop else src
            cur[idx] = cur[idx].lerp(src, 1. - decay) if exists(decay) else src
            return cur
        if exists(decay):
            return cur.lerp_(src, 1. - decay)
        if cur.shape == src.shape:
            return cur.copy_(src)
        return self.set(key, src.clone(), populated = populated, persistent = persistent)

    @torch.no_grad()
    def clear_(self, prefix: str = '', *, keys: str | Iterable[str] | None = None, pred: Callable[[str], bool] | None = None, indices: list[int] | None = None, pop_size: int | None = None, zero: bool = False):
        if exists(keys):
            key_set = {f'{prefix}{keys}'} if isinstance(keys, str) else {f'{prefix}{k}' for k in keys}
            matched = [k for k in self._populated if k in key_set]
        else:
            matched = [k for k in self._populated if k.startswith(prefix)]
        if exists(pred):
            matched = [k for k in matched if pred(k)]

        if not exists(indices) and not zero:
            for k in matched:
                self._populated.pop(k, None)
                self._repeat.discard(k)
                buf = _buf_key(k)
                self._buffers.pop(buf, None)
                self._non_persistent_buffers_set.discard(buf)
            return

        for k in matched:
            if not exists(indices):
                self[k].zero_()
            elif self.is_populated(k, pop_size):
                self[k][indices] = 0

    @torch.no_grad()
    def copy_from_(self, other: TensorStore, *, prefixes: tuple[str, ...] | None = None, indices: list[int] | None = None, pop_size: int | None = None):
        for k in self._populated:
            if exists(prefixes) and not k.startswith(prefixes):
                continue
            src = other.get(k)
            if not exists(src):
                continue
            dst = self[k]
            cast = src.detach().to(dst)
            if not exists(indices):
                dst.copy_(cast)
            elif self.is_populated(k, pop_size):
                dst[indices] = cast[indices]

    @torch.no_grad()
    def keep_(self, indices: list[int], old_pop: int):
        for k in self._populated:
            if self.is_populated(k, old_pop):
                buf = _buf_key(k)
                self._buffers[buf] = slice_pop(self._buffers[buf], indices)

    @torch.no_grad()
    def append_(self, old_pop: int):
        for k in self._populated:
            if self.is_populated(k, old_pop):
                buf = _buf_key(k)
                cur = self._buffers[buf]
                self._buffers[buf] = pad_pop(cur, cur[:1] if k in self._repeat else None)

    def clone(self, *, detach: bool = True, prefixes: tuple[str, ...] | None = None, shallow_prefixes: tuple[str, ...] = ()) -> TensorStore:
        out = TensorStore()
        for k, pop in self._populated.items():
            if exists(prefixes) and not k.startswith(prefixes):
                continue
            buf = _buf_key(k)
            t = self._buffers[buf]
            if not (shallow_prefixes and k.startswith(shallow_prefixes)):
                t = t.detach().clone() if detach else t.clone()
            out.set(k, t, populated = pop, persistent = buf not in self._non_persistent_buffers_set, append_repeat = k in self._repeat)
        return out

# prefix handle into PopulationParameters.store for optimizer state

class OptStore:
    def __init__(self, pp: PopulationParameters, name: str):
        self._pp = pp
        self.name = name
        self._prefix = f'opt/{name}/'
        self.slots: dict[str, SlotSpec] = {}
        self.aux: dict[tuple[str, str], AuxMeta] = {}

    def set(self, key: str, tensor: Tensor, *, populated: bool | None = None, append_repeat: bool = False) -> Tensor:
        return self._pp.store.set(self._prefix + key, tensor, populated = populated, append_repeat = append_repeat)

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

    def clear(self, pred: Callable[[str], bool] | None = None):
        if exists(pred):
            cut = len(self._prefix)
            self._pp.store.clear_(self._prefix, pred = lambda k: pred(k[cut:]))
            return
        self._pp.store.clear_(self._prefix)
        self.slots.clear()
        self.aux.clear()

    def alloc_slot(self, slot: str, param_names: Sequence[str] | None = None, *, shared: bool = False, dtypes: Mapping[str, torch.dtype] | None = None) -> OptStore:
        if slot in self.slots:
            return self
        pp = self._pp
        named = pp.named_parameters_dict()
        allowed = set(param_names) if exists(param_names) else None
        selected = [s for s in pp.trainable_specs if not exists(allowed) or s.name in allowed]
        pop_specs = [s for s in selected if not s.shared]
        sh_specs = [s for s in selected if s.shared]
        lead = None if (shared or not pp.is_population) else pp.pop_size

        for specs, key, pop_dim in ((pop_specs, f'slot/{slot}/pop', lead), (sh_specs, f'slot/{slot}/sh', None)):
            if not specs:
                continue
            p0 = named[specs[0].name]
            dt = dtypes.get(specs[0].name, p0.dtype) if exists(dtypes) else p0.dtype
            numel = sum(s.numel for s in specs)
            shape = (pop_dim, numel) if exists(pop_dim) else (numel,)
            self.set(key, torch.zeros(shape, device = p0.device, dtype = dt), populated = exists(pop_dim))

        self.slots[slot] = SlotSpec(
            shared = shared,
            pop_names = [s.name for s in pop_specs], pop_shapes = [s.shape for s in pop_specs],
            sh_names = [s.name for s in sh_specs], sh_shapes = [s.shape for s in sh_specs],
        )
        return self

    def alloc_aux(self, key: str, param: str, shape: tuple[int, ...], *, dtype: torch.dtype, device: torch.device, shared: bool = False, cpu_scalar: bool = False) -> Tensor:
        pair = (key, param)
        buf_key = f'aux/{key}/{param}'
        if pair in self.aux:
            return self[buf_key]
        pp = self._pp
        is_sh = shared or (pp.has_param(param) and pp.is_shared_param(param))
        is_pop = pp.is_population and not is_sh
        self.aux[pair] = AuxMeta(is_sh, cpu_scalar)
        full_shape = (pp.pop_size, *shape) if is_pop else shape
        return self.set(buf_key, torch.zeros(full_shape, device = device, dtype = dtype), populated = is_pop)

    def state_keys(self) -> list[str]:
        return list(dict.fromkeys((*self.slots, *(k for k, _ in self.aux))))

    def slot_views(self, slot: str, *, idx: int | None = None) -> dict[str, Tensor]:
        spec = self.slots[slot]
        views: dict[str, Tensor] = {}
        for suffix, names, shapes in (('pop', spec.pop_names, spec.pop_shapes), ('sh', spec.sh_names, spec.sh_shapes)):
            if names:
                buf = self[f'slot/{slot}/{suffix}']
                views.update(zip(names, unpack_tensors(buf, shapes, idx = idx if buf.ndim == 2 else None)))
        return views

    def all_views(self) -> dict[str, dict[str, Tensor]]:
        return {k: self.slot_views(k) for k in self.slots}

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

    def reset_members_(self, indices: list[int] | None):
        hp_prefix = f'{self._prefix}hp/'
        self._pp.store.clear_(self._prefix, pred = lambda k: not k.startswith(hp_prefix), indices = indices, pop_size = self._pp.pop_size, zero = True)

    def clone_handle(self, new_pp: PopulationParameters) -> OptStore:
        out = OptStore(new_pp, self.name)
        out.slots = deepcopy(self.slots)
        out.aux = deepcopy(self.aux)
        return out
