from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from typing import TYPE_CHECKING, Literal

import torch
from einops import pack, rearrange, unpack
from torch import Tensor, is_tensor
from torch.nn import Module, Parameter, ParameterList

from torch_population.store import (
    FunctionalOptState,
    OptStore,
    StoreDict,
    TensorSpec,
    TensorStore,
    default,
    exists,
    is_int_index,
    matches_pattern,
    materialize_from_template,
    named_buffer_persistence,
    pad_pop,
    reset_module_parameters,
    slice_pop,
    slice_spec,
    unpack_tensors,
)

if TYPE_CHECKING:
    from torch_population.optimizers import PopulationOptimizer

# unified colocated population parameters container

class PopulationParameters(Module):
    """Unified colocated storage for parameters, buffers, optimizer states, lanes, EMAs, and meta tensors."""

    def __init__(self):
        super().__init__()
        self.pop_size: int | None = None
        self.is_functional: bool = False
        self.specs: list[TensorSpec] = []
        self._spec_by_name: dict[str, TensorSpec] = {}
        self._state_dict_key_map: list[tuple[str, str, bool]] = []
        self._noop_net_ref: list[Module] = []
        self._template_ref: list[Module] = []

        self._trainable_flats = ParameterList()
        self._shared_trainable_flats = ParameterList()
        self._func_trainable_flats: list[Tensor] = []
        self._func_shared_flats: list[Tensor] = []

        self._frozen_store = TensorStore()
        self._buffer_store = TensorStore()

        self.opt_stores = StoreDict()
        self.ema_store = TensorStore()
        self.meta_store = TensorStore()
        self.lane_stores = StoreDict()
        self.substores = StoreDict()
        self.lanes: dict[str, Tensor] = {}

        self._functional_opt_states: dict[str, FunctionalOptState] = {}
        self._optimizers: dict[str, PopulationOptimizer] = {}
        self._opt_owners: dict[str, list[PopulationOptimizer]] = {}
        self._default_opt_name: str | None = None

        self._tracking_active_grads: bool = False
        self._active_grad_params: set[str] = set()

    # initialization & shell configuration

    def _set_shell(
        self,
        *,
        pop_size: int | None,
        is_functional: bool,
        noop_net: list[Module],
        template: list[Module],
        specs: list[TensorSpec],
        sd_map: list[tuple[str, str, bool]],
    ):
        self.pop_size = pop_size
        self.is_functional = is_functional
        self._noop_net_ref = list(noop_net)
        self._template_ref = list(template)
        self.specs = list(specs)
        self._spec_by_name = {s.name: s for s in self.specs}
        self._state_dict_key_map = list(sd_map)
        self.clear_grad_tracking()

    def _build_specs(self, ref: Module, *, shared = ()) -> tuple[list[TensorSpec], list[tuple[str, str, bool]]]:
        named_p = list(ref.named_parameters(remove_duplicate = True))
        groups = {(rg, sh): [{}, [], 0] for rg in (True, False) for sh in (False, True)}
        specs: list[TensorSpec] = []

        for name, p in named_p:
            rg = p.requires_grad
            sh = self.is_population and matches_pattern(name, shared)
            buckets, offsets, t_idx = groups[(rg, sh)]

            if p.dtype not in buckets:
                buckets[p.dtype] = len(buckets)
                offsets.append(0)

            b = buckets[p.dtype]
            specs.append(TensorSpec(name, tuple(p.shape), p.numel(), b, offsets[b], rg, t_idx if rg else None, sh))
            offsets[b] += p.numel()
            if rg:
                groups[(rg, sh)][2] = t_idx + 1

        spec_names = {s.name for s in specs}
        p_ids = {id(p): n for n, p in named_p}
        b_ids = {id(b): n for n, b in ref.named_buffers(remove_duplicate = True)}

        sd_map = []
        for k, t in ref.state_dict().items():
            if id(t) in p_ids:
                sd_map.append((k, p_ids[id(t)], True))
            elif id(t) in b_ids:
                sd_map.append((k, b_ids[id(t)], False))
            else:
                sd_map.append((k, k, k in spec_names))

        return specs, sd_map

    @staticmethod
    def _pack_group(
        specs: list[TensorSpec],
        all_params: list[dict[str, Tensor]],
        *,
        shared: bool,
        as_param: bool,
    ) -> list[Tensor]:
        num_buckets = max((s.bucket for s in specs), default = -1) + 1
        out: list[Tensor] = []

        for b in range(num_buckets):
            bs = [s for s in specs if s.bucket == b]
            stacked = [torch.stack([mp[s.name].detach() for mp in all_params], dim = 0) for s in bs]
            if shared:
                reduced = [t.mean(0) if t.is_floating_point() else t[0] for t in stacked]
                flat, _ = pack(reduced, '*')
            else:
                flat, _ = pack(stacked, 'p *')
            out.append(Parameter(flat, requires_grad = True) if as_param else flat)

        return out

    def init_noop(self, net: Module):
        self.pop_size = None
        specs, sd_map = self._build_specs(net)
        self._set_shell(
            pop_size = None,
            is_functional = False,
            noop_net = [net],
            template = [],
            specs = specs,
            sd_map = sd_map,
        )
        self._trainable_flats = ParameterList()
        self._shared_trainable_flats = ParameterList()
        self._func_trainable_flats = []
        self._func_shared_flats = []
        self._functional_opt_states = {}
        self._frozen_store.clear()
        self._buffer_store.clear()

    def init_from_modules(self, modules: list[Module], template: Module, *, shared = ()):
        ref = modules[0]
        all_p = [dict(m.named_parameters(remove_duplicate = True)) for m in modules]
        all_b = [dict(m.named_buffers(remove_duplicate = True)) for m in modules]
        ref_p, ref_b = all_p[0], all_b[0]

        for idx, (mp, mb) in enumerate(zip(all_p[1:], all_b[1:]), start = 1):
            if list(mp) != list(ref_p):
                raise ValueError(f'module {idx} parameter names do not match module 0')
            if list(mb) != list(ref_b):
                raise ValueError(f'module {idx} buffer names do not match module 0')
            for n, p in ref_p.items():
                if mp[n].shape != p.shape:
                    raise ValueError(f'parameter {n!r} shape mismatch: {p.shape} vs {mp[n].shape}')
            for n, b in ref_b.items():
                if mb[n].shape != b.shape:
                    raise ValueError(f'buffer {n!r} shape mismatch: {b.shape} vs {mb[n].shape}')

        self.pop_size = len(modules)
        specs, sd_map = self._build_specs(ref, shared = shared)
        self._set_shell(
            pop_size = len(modules),
            is_functional = False,
            noop_net = [],
            template = [template],
            specs = specs,
            sd_map = sd_map,
        )
        self._func_trainable_flats = []
        self._func_shared_flats = []
        self._functional_opt_states = {}

        pop_t = self._pack_group(self.pop_trainable_specs, all_p, shared = False, as_param = True)
        sh_t = self._pack_group(self.shared_trainable_specs, all_p, shared = True, as_param = True)
        self._trainable_flats = ParameterList(pop_t)
        self._shared_trainable_flats = ParameterList(sh_t)

        self._frozen_store.clear()
        pop_frozen = [s for s in self.frozen_specs if not s.shared]
        sh_frozen = [s for s in self.frozen_specs if s.shared]
        for b, t in enumerate(self._pack_group(pop_frozen, all_p, shared = False, as_param = False)):
            self._frozen_store.set(f'pop_{b}', t, populated = True)
        for b, t in enumerate(self._pack_group(sh_frozen, all_p, shared = True, as_param = False)):
            self._frozen_store.set(f'sh_{b}', t, populated = False)

        persistence = named_buffer_persistence(ref)
        self._buffer_store.clear()
        for name in ref_b:
            stacked = torch.stack([mb[name].detach().clone() for mb in all_b], dim = 0)
            self._buffer_store.set(name, stacked, populated = True, persistent = persistence.get(name, True))

    def _init_from_sliced(
        self,
        *,
        pop_size: int,
        template: Module,
        specs: list[TensorSpec],
        sd_map: list[tuple[str, str, bool]],
        trainable_flats: list[Parameter],
        shared_trainable_flats: list[Parameter],
        frozen_store: TensorStore,
        buffer_store: TensorStore,
    ):
        self._set_shell(
            pop_size = pop_size,
            is_functional = False,
            noop_net = [],
            template = [template],
            specs = specs,
            sd_map = sd_map,
        )
        self._trainable_flats = ParameterList(trainable_flats)
        self._shared_trainable_flats = ParameterList(shared_trainable_flats)
        self._func_trainable_flats = []
        self._func_shared_flats = []
        self._functional_opt_states = {}
        self._frozen_store = frozen_store
        self._buffer_store = buffer_store

    def _spawn_functional(
        self,
        trainable_flats: list[Tensor],
        opt_states: dict[str, FunctionalOptState],
        shared_flats: list[Tensor] | None = None,
    ) -> PopulationParameters:
        out = PopulationParameters()
        out._set_shell(
            pop_size = self.pop_size,
            is_functional = True,
            noop_net = self._noop_net_ref,
            template = self._template_ref,
            specs = self.specs,
            sd_map = self._state_dict_key_map,
        )
        out._func_trainable_flats = list(trainable_flats)
        out._func_shared_flats = list(default(shared_flats, self.param_leaves()[1]))
        out._frozen_store = self._frozen_store.clone_store()
        out._buffer_store = self._buffer_store.clone_store()
        out._functional_opt_states = dict(opt_states)
        out._opt_owners = {k: list(v) for k, v in self._opt_owners.items()}
        out._default_opt_name = self._default_opt_name
        return out

    # spec & store accessors

    @property
    def is_population(self) -> bool:
        return exists(self.pop_size)

    @property
    def has_shared_params(self) -> bool:
        return any(s.shared for s in self.specs)

    @property
    def trainable_specs(self) -> list[TensorSpec]:
        return [s for s in self.specs if s.requires_grad]

    @property
    def pop_trainable_specs(self) -> list[TensorSpec]:
        return [s for s in self.specs if s.requires_grad and not s.shared]

    @property
    def shared_trainable_specs(self) -> list[TensorSpec]:
        return [s for s in self.specs if s.requires_grad and s.shared]

    @property
    def frozen_specs(self) -> list[TensorSpec]:
        return [s for s in self.specs if not s.requires_grad]

    def has_param(self, name: str) -> bool:
        return name in self._spec_by_name

    def param_spec(self, name: str) -> TensorSpec:
        if name not in self._spec_by_name:
            raise KeyError(f'no parameter named {name!r}')
        return self._spec_by_name[name]

    def is_shared_param(self, name: str) -> bool:
        return self.param_spec(name).shared

    @property
    def num_params(self) -> int:
        return sum(s.numel for s in self.trainable_specs)

    @property
    def num_pop_params(self) -> int:
        return sum(s.numel for s in self.pop_trainable_specs)

    @property
    def param_names(self) -> list[str]:
        return [s.name for s in self.specs]

    @property
    def trainable_param_names(self) -> list[str]:
        return [s.name for s in self.trainable_specs]

    @property
    def buffer_names(self) -> list[str]:
        if not self.is_population:
            return [n for n, _ in self._noop_net_ref[0].named_buffers(remove_duplicate = True)]
        return self._buffer_store.keys()

    def param_leaves(self) -> tuple[list[Tensor], list[Tensor]]:
        if self.is_functional:
            return list(self._func_trainable_flats), list(self._func_shared_flats)
        return list(self._trainable_flats), list(self._shared_trainable_flats)

    @property
    def flat_params(self) -> Tensor:
        if not self.is_population:
            trainable = [p for p in self._noop_net_ref[0].parameters() if p.requires_grad]
            if not trainable:
                return torch.empty(0)
            if len({p.dtype for p in trainable}) > 1:
                raise ValueError('flat_params requires a single trainable parameter dtype')
            return pack(trainable, '*')[0]

        flats = self.param_leaves()[0]
        if not flats:
            return torch.empty((self.pop_size, 0))
        if len(flats) > 1:
            raise ValueError(f'Population has {len(flats)} trainable parameter dtypes')
        return flats[0]

    @property
    def flat_shared_params(self) -> Tensor | None:
        flats = self.param_leaves()[1]
        if not flats:
            return None
        if len(flats) > 1:
            raise ValueError(f'Population has {len(flats)} shared trainable parameter dtypes')
        return flats[0]

    @property
    def flat_grad(self) -> Tensor | None:
        if not self.is_population:
            trainable = [p for p in self._noop_net_ref[0].parameters() if p.requires_grad]
            if not any(exists(p.grad) for p in trainable):
                return None
            return pack([default(p.grad, torch.zeros_like(p)) for p in trainable], '*')[0]
        return self.flat_params.grad

    def _get_store(self, sdict: StoreDict, name: str, factory: Callable[[], Module], *, create: bool, label: str):
        if name not in sdict:
            if not create:
                raise KeyError(f'no {label} named {name!r}')
            sdict[name] = factory()
        return sdict[name]

    def register_opt_store(self, name: str) -> OptStore:
        return self._get_store(self.opt_stores, name, lambda: OptStore(self), create = True, label = 'optimizer store')

    def opt_store(self, name: str) -> OptStore:
        return self._get_store(self.opt_stores, name, lambda: OptStore(self), create = False, label = 'optimizer store')

    def lane_store(self, name: str, *, create: bool = True) -> TensorStore:
        return self._get_store(self.lane_stores, name, TensorStore, create = create, label = 'lane store')

    def substore(self, name: str, *, create: bool = True) -> TensorStore:
        return self._get_store(self.substores, name, TensorStore, create = create, label = 'substore')

    # generic optimizer state accessors (works for Adam, AdamW, SGD, Muon, custom)

    def _resolve_opt_key(self, opt: str | PopulationOptimizer | None = None) -> str | None:
        if not exists(opt):
            return self._default_opt_name
        name = opt if isinstance(opt, str) else opt.name
        stores = self._functional_opt_states if self.is_functional else self.opt_stores
        if name not in stores:
            raise KeyError(f'no optimizer named {name!r}')
        return name

    def _resolve_opt_owner(self, opt: str | PopulationOptimizer | None = None) -> PopulationOptimizer | None:
        key = self._resolve_opt_key(opt)
        if not exists(key):
            return None
        if key in self._optimizers:
            return self._optimizers[key]
        owners = self._opt_owners.get(key)
        return owners[0] if owners else None

    def opt_slot_flat(self, slot: str, opt: str | PopulationOptimizer | None = None) -> Tensor | None:
        key = self._resolve_opt_key(opt)
        if not exists(key):
            return None

        if self.is_functional:
            flats = self._functional_opt_states[key].slots.get(slot)
        else:
            store: OptStore = self.opt_stores[key]
            flats = store.slot_flats(slot) if store.has_slot(slot) else None

        if not flats:
            return None
        if len(flats) > 1:
            raise ValueError('multiple dtype buckets; use named_optimizer_state()')
        return flats[0]

    def opt_global(self, name: str, opt: str | PopulationOptimizer | None = None) -> Tensor | None:
        key = self._resolve_opt_key(opt)
        if not exists(key):
            return None
        if self.is_functional:
            return self._functional_opt_states[key].globals.get(name)
        store: OptStore = self.opt_stores[key]
        return store.global_tensor(name) if store.has_global(name) else None

    @property
    def flat_exp_avg(self) -> Tensor | None:
        return self.opt_slot_flat('m')

    @property
    def flat_exp_avg_sq(self) -> Tensor | None:
        return self.opt_slot_flat('v')

    @property
    def flat_steps(self) -> Tensor | None:
        return self.opt_global('steps')

    flat_m = flat_exp_avg
    flat_v = flat_exp_avg_sq

    # flat parameter-space packing / unpacking

    def pack_named(self, named: Mapping[str, Tensor], *, shared: bool = False) -> tuple[Tensor, Tensor | None]:
        pop_specs = self.pop_trainable_specs if self.is_population else self.trainable_specs
        sh_specs = self.shared_trainable_specs if self.is_population else []
        ref = default(next(iter(named.values()), None), self.flat_params)

        def _get(s: TensorSpec, *, collapse: bool) -> Tensor:
            v = named.get(s.name)
            if not exists(v):
                shape = s.shape if collapse else (self.pop_size, *s.shape)
                return torch.zeros(shape, device = ref.device, dtype = ref.dtype)
            if collapse and self.is_population and v.ndim == len(s.shape) + 1:
                return v.mean(0)
            return v

        collapse = not self.is_population or shared
        if pop_specs:
            pattern = '*' if collapse else 'p *'
            pop_flat, _ = pack([_get(s, collapse = collapse) for s in pop_specs], pattern)
        else:
            shape = (0,) if collapse else (self.pop_size, 0)
            pop_flat = torch.empty(shape, device = ref.device, dtype = ref.dtype)

        sh_flat = pack([_get(s, collapse = True) for s in sh_specs], '*')[0] if sh_specs else None
        return pop_flat, sh_flat

    def unpack_flat(self, flat: Tensor, sh_flat: Tensor | None = None) -> dict[str, Tensor]:
        pop_specs = self.pop_trainable_specs if self.is_population else self.trainable_specs
        pop_views = unpack_tensors(flat, [s.shape for s in pop_specs])
        parts = dict(zip((s.name for s in pop_specs), pop_views))

        if not exists(sh_flat):
            return parts

        sh_specs = self.shared_trainable_specs
        sh_views = unpack_tensors(sh_flat, [s.shape for s in sh_specs])
        parts.update(zip((s.name for s in sh_specs), sh_views))
        return {s.name: parts[s.name] for s in self.trainable_specs if s.name in parts}

    # colocated gradient lanes, EMAs, and meta tensors

    def _upsert_flat(
        self,
        store: TensorStore,
        key: str,
        value: Tensor,
        *,
        decay: float | None = None,
        members = None,
        shared: bool = False,
        source: str = '',
    ) -> Tensor:
        val = value.detach()
        if shared and self.is_population and val.ndim == 2:
            val = val.mean(0)
        idx = self.resolve_members(members) if (exists(members) and self.is_population) else None
        is_pop = self.is_population and not shared and val.ndim == 2
        return store.upsert_(key, val, decay = decay, idx = idx, pop_size = self.pop_size, source = source, populated = is_pop)

    def set_lane(self, name: str, grad: Tensor | Mapping[str, Tensor]) -> Tensor:
        flat = self.pack_named(grad)[0] if isinstance(grad, Mapping) else grad
        expected = self.num_pop_params if self.is_population else self.num_params
        if flat.shape[-1] != expected:
            raise ValueError(f'lane {name!r} last dim must be {expected}, got shape {tuple(flat.shape)}')
        self.lanes[name] = flat
        return flat

    def lane(self, name: str) -> Tensor:
        if name not in self.lanes:
            raise KeyError(f'no gradient lane named {name!r}')
        return self.lanes[name]

    def clear_lanes(self, names: str | Sequence[str] | None = None):
        if not exists(names):
            self.lanes.clear()
            return
        targets = (names,) if isinstance(names, str) else names
        for n in targets:
            self.lanes.pop(n, None)

    def update_lane_ema(
        self,
        lane: str,
        key: str = 'ema',
        decay: float = 0.9,
        *,
        members = None,
        shared: bool = False,
        source_tensor: Tensor | None = None,
    ) -> Tensor:
        src = source_tensor if exists(source_tensor) else self.lane(lane)
        return self._upsert_flat(self.lane_store(lane), key, src, decay = decay, members = members, shared = shared, source = lane)

    def update_lane_var(
        self,
        lane: str,
        key: str = 'var',
        decay: float = 0.99,
        *,
        members = None,
        shared: bool = False,
        source_tensor: Tensor | None = None,
    ) -> Tensor:
        g = (source_tensor if exists(source_tensor) else self.lane(lane)).detach()
        return self.update_lane_ema(lane, key = key, decay = decay, members = members, shared = shared, source_tensor = g * g)

    def lane_meta(self, lane: str) -> dict[str, Tensor]:
        return self.lane_stores[lane].as_dict() if lane in self.lane_stores else {}

    def clear_lane_meta(self, lane: str | None = None, key: str | None = None, *, members = None):
        idx = self.resolve_members(members) if exists(members) else None
        targets = [lane] if exists(lane) else list(self.lane_stores.keys())
        for lname in targets:
            if lname in self.lane_stores:
                self.lane_stores[lname].clear_keys_(key, indices = idx, pop_size = self.pop_size)

    def update_ema(
        self,
        tag: str,
        source: str | Tensor,
        decay: float = 0.99,
        *,
        members = None,
        shared: bool = False,
    ) -> Tensor:
        if is_tensor(source):
            src_t, src_key = source, ''
        elif source == 'params':
            src_t, src_key = self.flat_params, 'params'
        elif source.startswith('grad.'):
            src_t, src_key = self.lane(source[5:]), source
        elif source in self.lanes:
            src_t, src_key = self.lanes[source], f'grad.{source}'
        else:
            raise KeyError(f"unknown EMA source {source!r}; expected 'params', 'grad.<lane>', a lane name, or a Tensor")
        return self._upsert_flat(self.ema_store, tag, src_t, decay = decay, members = members, shared = shared, source = src_key)

    def ema(self, tag: str) -> Tensor:
        if tag not in self.ema_store:
            raise KeyError(f'no EMA named {tag!r}')
        return self.ema_store[tag]

    @property
    def emas(self) -> dict[str, Tensor]:
        return self.ema_store.as_dict()

    def clear_ema(self, tag: str | None = None, *, members = None):
        idx = self.resolve_members(members) if exists(members) else None
        self.ema_store.clear_keys_(tag, indices = idx, pop_size = self.pop_size)

    def set_meta(self, name: str, value: Tensor, *, members = None) -> Tensor:
        idx = self.resolve_members(members) if (exists(members) and self.is_population) else None
        return self.meta_store.upsert_(name, value, idx = idx, pop_size = self.pop_size)

    def ema_meta(self, name: str, value: Tensor, decay: float = 0.95, *, members = None) -> Tensor:
        self.set_meta(name, value, members = members)
        idx = self.resolve_members(members) if (exists(members) and self.is_population) else None
        return self.meta_store.upsert_(f'ema.{name}', value, decay = decay, idx = idx, pop_size = self.pop_size)

    @property
    def meta(self) -> dict[str, Tensor]:
        return self.meta_store.as_dict()

    def clear_meta(self, name: str | None = None, *, members = None):
        idx = self.resolve_members(members) if exists(members) else None
        keys = {name, f'ema.{name}'} if exists(name) else None
        self.meta_store.clear_keys_(keys, indices = idx, pop_size = self.pop_size)

    # zero-copy structured views & direct param sync

    def clear_grad_tracking(self):
        self._tracking_active_grads = False
        self._active_grad_params.clear()

    def _param_flat_for(self, spec: TensorSpec) -> Tensor:
        t_flats, st_flats = self.param_leaves()
        if spec.shared:
            return st_flats[spec.bucket] if spec.requires_grad else self._frozen_store[f'sh_{spec.bucket}']
        return t_flats[spec.bucket] if spec.requires_grad else self._frozen_store[f'pop_{spec.bucket}']

    def copy_param_from_(self, name: str, value: Tensor, *, index: int | None = None, grad: Tensor | None = None):
        with torch.no_grad():
            if not self.is_population:
                dst = dict(self._noop_net_ref[0].named_parameters(remove_duplicate = True))[name]
                dst.copy_(value.detach().to(dst))
                if exists(grad):
                    dst.grad = grad.detach().clone()
                return

            spec = self.param_spec(name)
            flat = self._param_flat_for(spec)
            use_idx = index if not spec.shared else None
            dst = slice_spec(spec, flat, idx = use_idx, pop_size = self.pop_size)
            dst.copy_(value.detach().to(dst))

            if exists(grad) and spec.requires_grad:
                if not exists(flat.grad):
                    flat.grad = torch.zeros_like(flat)
                gdst = slice_spec(spec, flat.grad, idx = use_idx, pop_size = self.pop_size)
                gdst.copy_(grad.detach().to(gdst))
                self._active_grad_params.add(name)

    def named_parameters_dict(self, *, track_grads: bool = False) -> dict[str, Tensor]:
        if not self.is_population:
            return dict(self._noop_net_ref[0].named_parameters(remove_duplicate = True))

        should_hook = track_grads and torch.is_grad_enabled() and not self.is_functional
        if should_hook:
            self._tracking_active_grads = True

        out = {}
        for s in self.specs:
            view = slice_spec(s, self._param_flat_for(s), pop_size = self.pop_size)
            if should_hook and s.requires_grad and view.requires_grad:
                view.register_hook(lambda _g, n = s.name: self._active_grad_params.add(n))
            out[s.name] = view
        return out

    def batched_parameters_dict(self, *, members = None, track_grads: bool = False) -> dict[str, Tensor]:
        named = self.named_parameters_dict(track_grads = track_grads)
        if not self.is_population:
            return named

        idx = self.resolve_members(members) if exists(members) else None
        n = len(idx) if exists(idx) else self.pop_size
        if not exists(idx) and not self.has_shared_params:
            return named

        out = {}
        for s in self.specs:
            t = named[s.name]
            if s.shared:
                out[s.name] = t.unsqueeze(0).expand(n, *s.shape)
            else:
                out[s.name] = t[idx] if exists(idx) else t
        return out

    def batched_buffers_dict(self, *, members = None) -> dict[str, Tensor]:
        bufs = self.named_buffers_dict()
        if not self.is_population or not exists(members):
            return bufs
        idx = self.resolve_members(members)
        return {n: b[idx] for n, b in bufs.items()}

    def named_grads_dict(self) -> dict[str, Tensor | None]:
        if not self.is_population:
            return {n: p.grad for n, p in self._noop_net_ref[0].named_parameters(remove_duplicate = True)}

        t_flats, st_flats = self.param_leaves()
        out = {}
        for s in self.specs:
            gf = (st_flats[s.bucket].grad if s.shared else t_flats[s.bucket].grad) if s.requires_grad else None
            if not exists(gf) or (self._tracking_active_grads and s.name not in self._active_grad_params):
                out[s.name] = None
            else:
                out[s.name] = slice_spec(s, gf, pop_size = self.pop_size)
        return out

    def named_buffers_dict(self) -> dict[str, Tensor]:
        if not self.is_population:
            return dict(self._noop_net_ref[0].named_buffers(remove_duplicate = True))
        return self._buffer_store.as_dict()

    def named_optimizer_state(self, opt: str | PopulationOptimizer | None = None) -> dict[str, dict[str, Tensor]] | None:
        owner = self._resolve_opt_owner(opt)
        return owner.named_state(self) if exists(owner) else None

    def optimizer_state_dict(self, opt: str | PopulationOptimizer | None = None) -> dict | None:
        owner = self._resolve_opt_owner(opt)
        return owner.state_dict_for(self) if exists(owner) else None

    def _build_state_dict(self, named_p: Mapping[str, Tensor], named_b: Mapping[str, Tensor]) -> dict[str, Tensor]:
        out = {}
        for sd_k, canon, is_param in self._state_dict_key_map:
            table = named_p if is_param else named_b
            if canon in table:
                out[sd_k] = table[canon]
        return out

    def as_dict(self, opt: str | PopulationOptimizer | None = None) -> dict:
        named_p = self.named_parameters_dict()
        named_b = self.named_buffers_dict()
        return {
            'parameters': list(named_p.values()),
            'named_parameters': named_p,
            'buffers': list(named_b.values()),
            'named_buffers': named_b,
            'optimizer_state': self.optimizer_state_dict(opt),
            'named_optimizer_state': self.named_optimizer_state(opt),
        }

    # indexing & PyTorch-compatible single-individual dicts

    def _normalize_index(self, index: int | Tensor) -> int:
        idx = int(index.item()) if is_tensor(index) else int(index)
        size = self.pop_size if self.is_population else 1
        idx = idx + size if idx < 0 else idx
        if not (0 <= idx < size):
            raise IndexError(f'individual index {index} out of range for population of size {size}')
        return idx

    def resolve_members(self, selector) -> list[int]:
        size = self.pop_size if self.is_population else 1
        if is_int_index(selector):
            return [self._normalize_index(selector)]
        if isinstance(selector, slice):
            return list(range(size))[selector]
        if is_tensor(selector):
            if selector.dtype == torch.bool:
                if selector.ndim != 1 or selector.numel() != size:
                    raise ValueError(f'boolean mask must have shape ({size},), got {tuple(selector.shape)}')
                return selector.nonzero(as_tuple = False).flatten().tolist()
            if selector.ndim != 1 or selector.is_floating_point():
                raise IndexError(f'index tensor must be 1D integer or bool, got shape {tuple(selector.shape)}, dtype {selector.dtype}')
            return [self._normalize_index(int(i)) for i in selector.tolist()]
        if isinstance(selector, Sequence) and not isinstance(selector, (str, bytes)):
            return [self._normalize_index(i) for i in selector]
        raise TypeError(f'unsupported selector type {type(selector).__name__}')

    def individual(
        self,
        index: int = 0,
        *,
        opt: str | PopulationOptimizer | None = None,
        clone: bool = False,
    ) -> dict:
        idx = self._normalize_index(index)
        maybe = (lambda t: t.detach().clone()) if clone else (lambda t: t)

        if not self.is_population:
            net = self._noop_net_ref[0]
            named_p = {n: maybe(p) for n, p in net.named_parameters(remove_duplicate = True)}
            named_b = {n: maybe(b) for n, b in net.named_buffers(remove_duplicate = True)}
        else:
            named_p = {s.name: maybe(slice_spec(s, self._param_flat_for(s), idx = idx)) for s in self.specs}
            named_b = {n: maybe(b[idx]) for n, b in self._buffer_store.items()}

        owner = self._resolve_opt_owner(opt)
        if exists(owner):
            named_opt, opt_sd = owner.individual_state(self, idx if self.is_population else None, clone = clone)
        else:
            named_opt, opt_sd = None, None

        out = {
            'parameters': list(named_p.values()),
            'named_parameters': named_p,
            'buffers': list(named_b.values()),
            'named_buffers': named_b,
            'state_dict': self._build_state_dict(named_p, named_b),
            'optimizer_state': opt_sd,
            'named_optimizer_state': named_opt,
        }

        if self.lanes:
            out['lanes'] = {
                k: self.unpack_flat(maybe(v[idx] if (self.is_population and v.ndim == 2) else v))
                for k, v in self.lanes.items()
            }
        if len(self.ema_store):
            out['emas'] = {
                k: self.unpack_flat(maybe(v[idx] if (self.is_population and v.ndim == 2) else v))
                for k, v in self.ema_store.items()
            }
        if len(self.meta_store):
            out['meta'] = {
                k: maybe(v[idx] if self.meta_store.is_populated(k, self.pop_size) else v)
                for k, v in self.meta_store.items()
            }
        return out

    def load_individual_dict_(
        self,
        index: int,
        data: Mapping,
        *,
        opt: str | PopulationOptimizer | None = None,
    ) -> PopulationParameters:
        idx = self._normalize_index(index)
        sd = data.get('state_dict', {})
        src_p = data.get('named_parameters') or ({s.name: sd[s.name] for s in self.specs if s.name in sd} if sd else None)
        src_b = data.get('named_buffers') or ({n: sd[n] for n in self.buffer_names if n in sd} if sd else None)

        with torch.no_grad():
            if exists(src_p):
                for s in self.specs:
                    if s.name in src_p:
                        if tuple(src_p[s.name].shape) != s.shape:
                            raise ValueError(f'parameter {s.name!r} shape mismatch: {s.shape} vs {tuple(src_p[s.name].shape)}')
                        self.copy_param_from_(s.name, src_p[s.name], index = idx)

            if exists(src_b):
                for n, dst in self.named_buffers_dict().items():
                    if n in src_b:
                        target = dst[idx] if self.is_population else dst
                        if tuple(src_b[n].shape) != tuple(target.shape):
                            raise ValueError(f'buffer {n!r} shape mismatch: {tuple(target.shape)} vs {tuple(src_b[n].shape)}')
                        target.copy_(src_b[n].detach().to(target))

            if exists(data.get('named_optimizer_state')) or exists(data.get('optimizer_state')):
                owner = self._resolve_opt_owner(opt)
                if not exists(owner):
                    raise ValueError('no attached optimizer on PopulationParameters to load optimizer_state into')
                owner.load_individual_state_dict_(self, idx, data)

        return self

    def __len__(self) -> int:
        return self.pop_size if self.is_population else 1

    def __getitem__(self, index: int) -> dict:
        return self.individual(index)

    def __setitem__(self, index: int, value: Mapping):
        if not isinstance(value, Mapping):
            raise TypeError(f'PopulationParameters item assignment expects a dict/Mapping, got {type(value).__name__}')
        self.load_individual_dict_(index, value)

    # lifecycle ops (DiscoRL / TBPTT / EMA / PBT dynamic resizing)

    def _dynamic_stores(self) -> tuple[TensorStore, ...]:
        return (
            *self.opt_stores.values(),
            self.ema_store,
            self.meta_store,
            *self.lane_stores.values(),
            *self.substores.values(),
        )

    def detach(self) -> PopulationParameters:
        if not self.is_population:
            raise ValueError('detach() is intended for populated PopulationParameters')
        if not self.is_functional:
            return self.clone()
        return self._spawn_functional(
            [f.detach().clone().requires_grad_(True) for f in self._func_trainable_flats],
            {n: st.map_tensors(lambda t: t.detach().clone()) for n, st in self._functional_opt_states.items()},
            shared_flats = [f.detach().clone().requires_grad_(True) for f in self._func_shared_flats],
        )

    def clone(self) -> PopulationParameters:
        if not self.is_population:
            raise ValueError('clone() requires a populated PopulationParameters')
        if self.is_functional:
            return self._spawn_functional(
                [f.clone() for f in self._func_trainable_flats],
                {n: st.map_tensors(lambda t: t.clone()) for n, st in self._functional_opt_states.items()},
                shared_flats = [f.clone() for f in self._func_shared_flats],
            )

        out = PopulationParameters()
        out._init_from_sliced(
            pop_size = self.pop_size,
            template = deepcopy(self._template_ref[0]),
            specs = self.specs,
            sd_map = self._state_dict_key_map,
            trainable_flats = [Parameter(f.detach().clone(), requires_grad = True) for f in self._trainable_flats],
            shared_trainable_flats = [Parameter(f.detach().clone(), requires_grad = True) for f in self._shared_trainable_flats],
            frozen_store = self._frozen_store.clone_store(),
            buffer_store = self._buffer_store.clone_store(),
        )
        for name, store in self.opt_stores.items():
            out.opt_stores[name] = store.clone_for(out)
        out._opt_owners = {k: list(v) for k, v in self._opt_owners.items()}
        out._default_opt_name = self._default_opt_name
        return out

    def mean_dict(self, weights: Tensor | Sequence[float] | None = None) -> dict:
        if not self.is_population:
            return self.individual(0, clone = True)

        w = None
        if exists(weights):
            w = weights.detach().float() if is_tensor(weights) else torch.tensor([float(v) for v in weights], dtype = torch.float32)
            if w.ndim != 1 or w.shape[0] != self.pop_size:
                raise ValueError(f'weights must have shape ({self.pop_size},), got {tuple(w.shape)}')

        def _red(t: Tensor) -> Tensor:
            td = t.detach()
            if not td.is_floating_point():
                return td[0].clone()
            if not exists(w):
                return td.mean(0)
            flat, ps = pack([td], 'p *')
            return unpack(w.to(td) @ flat, ps, '*')[0]

        named_p = {}
        for s in self.specs:
            view = slice_spec(s, self._param_flat_for(s), pop_size = self.pop_size)
            named_p[s.name] = view.detach().clone() if s.shared else _red(view)

        named_b = {n: _red(b) for n, b in self._buffer_store.items()}
        return {
            'parameters': list(named_p.values()),
            'named_parameters': named_p,
            'buffers': list(named_b.values()),
            'named_buffers': named_b,
            'state_dict': self._build_state_dict(named_p, named_b),
        }

    def lerp_(self, other: PopulationParameters, weight: float) -> PopulationParameters:
        with torch.no_grad():
            t_d, st_d = self.param_leaves()
            t_s, st_s = other.param_leaves()
            pairs = [
                *zip(t_d, t_s),
                *zip(st_d, st_s),
                *((self._frozen_store[k], other._frozen_store[k]) for k in self._frozen_store if k in other._frozen_store),
                *((self._buffer_store[k], other._buffer_store[k]) for k in self._buffer_store if k in other._buffer_store),
            ]
            for dst, src in pairs:
                if dst.is_floating_point():
                    dst.lerp_(src.detach().to(dst), weight)
        return self

    def copy_from_(self, other: PopulationParameters, *, members = None, include_opt: bool = True) -> PopulationParameters:
        idx = self.resolve_members(members) if exists(members) else None
        if exists(idx) and not idx:
            return self

        with torch.no_grad():
            t_d, st_d = self.param_leaves()
            t_s, st_s = other.param_leaves()
            for dst, src in zip(t_d, t_s):
                val = src.detach().to(dst)
                if exists(idx) and self.is_population:
                    dst[idx] = val[idx]
                else:
                    dst.copy_(val)

            if not exists(idx):
                for dst, src in zip(st_d, st_s):
                    dst.copy_(src.detach().to(dst))

            self._frozen_store.copy_from_(other._frozen_store, indices = idx, pop_size = self.pop_size)
            self._buffer_store.copy_from_(other._buffer_store, indices = idx, pop_size = self.pop_size)

            if include_opt:
                for opt_name, dst_store in self.opt_stores.items():
                    if other.is_functional and opt_name in other._functional_opt_states:
                        dst_store.copy_from_functional_(other._functional_opt_states[opt_name], indices = idx, pop_size = self.pop_size)
                    elif not other.is_functional and opt_name in other.opt_stores:
                        dst_store.copy_from_(other.opt_stores[opt_name], indices = idx, pop_size = self.pop_size)
        return self

    def reset_(
        self,
        selector = None,
        *,
        init: Literal['reinit'] | Callable[[Module], None] = 'reinit',
        reset_opt: bool = True,
        reset_buffers: bool = True,
    ) -> PopulationParameters:
        if not self.is_population:
            net = self._noop_net_ref[0]
            init(net) if callable(init) else reset_module_parameters(net)
            if reset_opt:
                for store in self.opt_stores.values():
                    store.reset_members_(None)
            return self

        indices = list(range(self.pop_size)) if not exists(selector) else self.resolve_members(selector)
        if not indices:
            return self

        with torch.no_grad():
            for idx in indices:
                ind = self.individual(idx, clone = True)
                fresh = materialize_from_template(self._template_ref[0], ind['named_parameters'], ind['named_buffers'], training = True)
                init(fresh) if callable(init) else reset_module_parameters(fresh)

                fresh_p = dict(fresh.named_parameters(remove_duplicate = True))
                for s in self.specs:
                    if not s.shared:
                        self.copy_param_from_(s.name, fresh_p[s.name], index = idx)

                if reset_buffers:
                    fresh_b = dict(fresh.named_buffers(remove_duplicate = True))
                    for n, dst in self._buffer_store.items():
                        dst[idx].copy_(fresh_b[n].detach().to(dst))

            if reset_opt:
                if self.is_functional:
                    for fstate in self._functional_opt_states.values():
                        fstate.reset_members_(indices)
                else:
                    for store in self.opt_stores.values():
                        store.reset_members_(indices)

            t_flats = self.param_leaves()[0]
            for tag, ebuf in self.ema_store.items():
                if not self.ema_store.is_populated(tag, self.pop_size):
                    continue
                if self.ema_store.source_of(tag) == 'params' and len(t_flats) == 1:
                    ebuf[indices] = t_flats[0][indices].detach().to(ebuf)
                else:
                    ebuf[indices] = 0.

            self.meta_store.reset_(indices, pop_size = self.pop_size)
            for lstore in self.lane_stores.values():
                lstore.reset_(indices, pop_size = self.pop_size)
        return self

    def _extract_source(self, source) -> tuple[dict[str, Tensor], dict[str, Tensor] | None]:
        if is_int_index(source):
            ind = self.individual(self._normalize_index(source), clone = True)
            return ind['named_parameters'], ind['named_buffers']
        if isinstance(source, Module):
            return dict(source.named_parameters(remove_duplicate = True)), dict(source.named_buffers(remove_duplicate = True))
        if isinstance(source, Mapping):
            if 'named_parameters' in source:
                return dict(source['named_parameters']), source.get('named_buffers')
            return dict(source), None
        if hasattr(source, 'items'):
            return dict(source.items()), None
        raise TypeError(f'source must be a member index, nn.Module, or parameter mapping, got {type(source).__name__}')

    def replace_(
        self,
        index: int,
        source,
        *,
        reset_opt: bool = True,
        reset_ema: bool = False,
        reset_meta: bool = False,
        reset_lanes: bool = False,
    ) -> PopulationParameters:
        if not self.is_population:
            raise ValueError('replace_() requires a populated PopulationParameters')

        idx = self._normalize_index(index)
        src_p, src_b = self._extract_source(source)

        with torch.no_grad():
            for s in self.specs:
                if not s.shared and s.name in src_p:
                    self.copy_param_from_(s.name, src_p[s.name], index = idx)

            if exists(src_b):
                for n, dst in self._buffer_store.items():
                    if n in src_b:
                        dst[idx].copy_(src_b[n].detach().to(dst))

            if reset_opt:
                for store in self.opt_stores.values():
                    store.reset_members_([idx])
            if reset_ema:
                self.clear_ema(members = [idx])
            if reset_meta:
                self.clear_meta(members = [idx])
            if reset_lanes:
                for lf in self.lanes.values():
                    if lf.ndim >= 1 and lf.shape[0] == self.pop_size:
                        lf[idx] = 0.
                self.clear_lane_meta(members = [idx])
        return self

    def keep_(self, members) -> PopulationParameters:
        if not self.is_population:
            raise ValueError('keep_() requires a populated PopulationParameters')

        keep_idx = list(dict.fromkeys(self.resolve_members(members)))
        if not keep_idx:
            raise ValueError('cannot delete all members of a population')

        old_pop = self.pop_size
        with torch.no_grad():
            new_trainable = []
            for f in self._trainable_flats:
                p = Parameter(slice_pop(f, keep_idx), requires_grad = True)
                if exists(f.grad):
                    p.grad = slice_pop(f.grad, keep_idx)
                new_trainable.append(p)
            self._trainable_flats = ParameterList(new_trainable)

            for store in (self._frozen_store, self._buffer_store, *self._dynamic_stores()):
                store.keep_(keep_idx, old_pop)

            for n, lf in list(self.lanes.items()):
                if lf.ndim >= 1 and lf.shape[0] == old_pop:
                    self.lanes[n] = slice_pop(lf, keep_idx)

        self.pop_size = len(keep_idx)
        self.clear_grad_tracking()
        for att in self._optimizers.values():
            att._sync_optimizer_state()
        return self

    def delete_(self, members) -> PopulationParameters:
        if not self.is_population:
            raise ValueError('delete_() requires a populated PopulationParameters')
        drop = set(self.resolve_members(members))
        keep_idx = [i for i in range(self.pop_size) if i not in drop]
        if not keep_idx:
            raise ValueError('cannot delete all members of a population')
        return self.keep_(keep_idx)

    def append_(self, source) -> int:
        if not self.is_population:
            raise ValueError('append_() requires a populated PopulationParameters')

        src_p, src_b = self._extract_source(source)
        old_pop = self.pop_size

        def _row_for(specs: list[TensorSpec], bucket: int, flat: Tensor) -> Tensor:
            bs = [s for s in specs if s.bucket == bucket]
            for s in bs:
                if s.name not in src_p:
                    raise KeyError(f'source is missing parameter {s.name!r}')
            row, _ = pack([src_p[s.name].detach().to(flat) for s in bs], '*')
            return rearrange(row, 'd -> 1 d')

        with torch.no_grad():
            self._trainable_flats = ParameterList([
                Parameter(pad_pop(f, _row_for(self.pop_trainable_specs, b, f)), requires_grad = True)
                for b, f in enumerate(self._trainable_flats)
            ])

            pop_frozen = [s for s in self.frozen_specs if not s.shared]
            for b in range(max((s.bucket for s in pop_frozen), default = -1) + 1):
                k = f'pop_{b}'
                row = _row_for(pop_frozen, b, self._frozen_store[k])
                self._frozen_store.set(k, pad_pop(self._frozen_store[k], row), populated = True)

            for name, buf in list(self._buffer_store.items()):
                pad = src_b[name].unsqueeze(0) if (exists(src_b) and name in src_b) else buf[:1]
                persistent = self._buffer_store._meta[name].persistent
                self._buffer_store.set(name, pad_pop(buf, pad), populated = True, persistent = persistent)

            for store in self._dynamic_stores():
                store.append_(old_pop)

            for n, lf in list(self.lanes.items()):
                if lf.ndim >= 1 and lf.shape[0] == old_pop:
                    self.lanes[n] = pad_pop(lf)

        self.pop_size = old_pop + 1
        self.clear_grad_tracking()
        for att in self._optimizers.values():
            att._sync_optimizer_state()
        return old_pop

    def extra_repr(self) -> str:
        if not self.is_population:
            return f'noop, num_params={self.num_params}'
        mode = 'functional' if self.is_functional else 'backed'
        return f'pop_size={self.pop_size}, num_params={self.num_params}, mode={mode}'
