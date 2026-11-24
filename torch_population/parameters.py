from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from copy import deepcopy
from typing import TYPE_CHECKING, Literal

import torch
from einops import pack
from torch import Tensor, is_tensor
from torch.autograd import grad as torch_grad
from torch.nn import Module, Parameter

from torch_population.store import (
    OptStore,
    TensorSpec,
    TensorStore,
    clone_template,
    default,
    exists,
    init_module,
    is_int_index,
    matches_pattern,
    normalize_index,
    pad_pop,
    resolve_indices,
    slice_pop,
    slice_spec,
    unpack_tensors,
)

if TYPE_CHECKING:
    from torch_population.optimizer import PopulationOptimizer

# helpers

def validate_module_maps(kind: str, maps: list[dict[str, Tensor]]):
    ref, ref_names = maps[0], list(maps[0])
    for i, cur in enumerate(maps[1:], start = 1):
        if list(cur) != ref_names:
            raise ValueError(f'module {i} {kind} names do not match module 0')
        for name, tensor in ref.items():
            if cur[name].shape != tensor.shape:
                raise ValueError(f'{kind} {name!r} shape mismatch: {tensor.shape} vs {cur[name].shape}')

def pack_param_group(specs: list[TensorSpec], module_params: list[dict[str, Tensor]], *, is_shared: bool) -> Parameter | None:
    if not specs:
        return None
    if len({module_params[0][s.name].dtype for s in specs}) > 1:
        raise ValueError('trainable parameters in a Population must share the same dtype')

    stacked = [torch.stack([mp[s.name].detach() for mp in module_params]) for s in specs]
    if is_shared:
        packed, _ = pack([t.mean(dim = 0) if t.is_floating_point() else t[0] for t in stacked], '*')
    else:
        packed, _ = pack(stacked, 'p *')
    return Parameter(packed, requires_grad = True)

def copy_tensor_(dst: Tensor, src: Tensor, kind: str, name: str, *, check_shapes: bool = False):
    if check_shapes and src.shape != dst.shape:
        raise ValueError(f'{kind} {name!r} shape mismatch: {tuple(dst.shape)} vs {tuple(src.shape)}')
    dst.copy_(src.detach().to(dst))

# parameter layout schema

class _Schema:
    def __init__(self, ref: Module | None = None, shared = ()):
        self.specs: list[TensorSpec] = []
        self.state_dict_map: list[tuple[str, str, bool]] = []

        if exists(ref):
            named_params = list(ref.named_parameters())
            pop_offset = shared_offset = trainable_idx = 0

            for name, param in named_params:
                is_shared = matches_pattern(name, shared)
                if not param.requires_grad:
                    self.specs.append(TensorSpec(name, tuple(param.shape), param.numel(), 0, False, None, is_shared))
                    continue

                offset = shared_offset if is_shared else pop_offset
                self.specs.append(TensorSpec(name, tuple(param.shape), param.numel(), offset, True, trainable_idx, is_shared))
                if is_shared:
                    shared_offset += param.numel()
                else:
                    pop_offset += param.numel()
                trainable_idx += 1

            by_id = {id(p): (name, True) for name, p in named_params}
            by_id.update({id(b): (name, False) for name, b in ref.named_buffers()})
            param_names = {s.name for s in self.specs}
            self.state_dict_map = [
                (sd_key, *by_id.get(id(t), (sd_key, sd_key in param_names)))
                for sd_key, t in ref.state_dict().items()
            ]

        self.by_name = {s.name: s for s in self.specs}
        self.param_names = [s.name for s in self.specs]
        self.trainable = [s for s in self.specs if s.requires_grad]
        self.pop_trainable = [s for s in self.trainable if not s.shared]
        self.shared_trainable = [s for s in self.trainable if s.shared]
        self.has_shared = any(s.shared for s in self.specs)

# unified colocated population parameter container

class PopulationParameters(Module):
    def __init__(self):
        super().__init__()
        self.register_parameter('_pop_param', None)
        self.register_parameter('_shared_param', None)
        self.store = TensorStore()
        self.opt_stores: dict[str, OptStore] = {}
        self._optimizers: dict[str, PopulationOptimizer] = {}
        self._default_opt_name: str | None = None
        self._active_grads: set[str] | None = None
        self.pop_size: int | None = None
        object.__setattr__(self, '_noop_net', None)
        object.__setattr__(self, '_template', None)
        object.__setattr__(self, '_factory', None)
        self._schema = _Schema()
        self._func_leaves: tuple[Tensor | None, Tensor | None] | None = None

    # initialization & cloning

    def init_noop(self, net: Module, *, factory: Callable[[], Module] | None = None):
        self.pop_size = None
        object.__setattr__(self, '_noop_net', net)
        object.__setattr__(self, '_template', None)
        object.__setattr__(self, '_factory', factory)
        self._schema = _Schema(net)
        self._func_leaves = None
        self._active_grads = None

    def init_from_modules(self, modules: list[Module], template: Module, *, shared = (), factory: Callable[[], Module] | None = None):
        ref = modules[0]
        module_params = [dict(m.named_parameters()) for m in modules]
        module_buffers = [dict(m.named_buffers()) for m in modules]
        validate_module_maps('parameter', module_params)
        validate_module_maps('buffer', module_buffers)

        schema = _Schema(ref, shared)
        self.pop_size = len(modules)
        object.__setattr__(self, '_noop_net', None)
        object.__setattr__(self, '_template', template)
        if exists(factory):
            object.__setattr__(self, '_factory', factory)
        self._schema = schema
        self._func_leaves = None
        self._active_grads = None

        self._pop_param = pack_param_group(schema.pop_trainable, module_params, is_shared = False)
        self._shared_param = pack_param_group(schema.shared_trainable, module_params, is_shared = True)

        self.store.remove_('frozen/')
        for spec in schema.specs:
            if spec.requires_grad:
                continue
            stacked = torch.stack([mp[spec.name].detach().clone() for mp in module_params])
            if spec.shared:
                stacked = stacked.mean(dim = 0) if stacked.is_floating_point() else stacked[0]
            self.store.set(f'frozen/{spec.name}', stacked, populated = not spec.shared)

        sd_keys = set(ref.state_dict())
        self.store.remove_('buffer/')
        for name in module_buffers[0]:
            stacked = torch.stack([mb[name].detach().clone() for mb in module_buffers])
            self.store.set(f'buffer/{name}', stacked, populated = True, persistent = name in sd_keys, append_repeat = True)

    def _spawn(
        self,
        store: TensorStore,
        pop_flat: Tensor | None,
        shared_flat: Tensor | None,
        *,
        pop_size: int | None = None,
        template: Module | None = None,
        functional: bool = False,
        clone_opts: bool = True,
    ) -> PopulationParameters:
        out = PopulationParameters()
        out.pop_size = default(pop_size, self.pop_size)
        object.__setattr__(out, '_noop_net', self._noop_net)
        object.__setattr__(out, '_template', default(template, self._template))
        object.__setattr__(out, '_factory', self._factory)
        out._schema = self._schema
        out.store = store

        if functional:
            out._func_leaves = (pop_flat, shared_flat)
        else:
            out._pop_param = Parameter(pop_flat, requires_grad = True) if exists(pop_flat) else None
            out._shared_param = Parameter(shared_flat, requires_grad = True) if exists(shared_flat) else None

        if clone_opts:
            out.opt_stores = {k: v.clone_handle(out) for k, v in self.opt_stores.items()}
            out._optimizers = dict(self._optimizers)
            out._default_opt_name = self._default_opt_name
        return out

    def _store_prefixes(self, include_opt: bool, include_buffers: bool = True) -> tuple[str, ...]:
        prefixes = ('frozen/',)
        if include_buffers:
            prefixes += ('buffer/',)
        if include_opt:
            prefixes += ('opt/',)
        return prefixes

    def sliced(self, indices: list[int] | None = None, *, clone_opts: bool = False) -> PopulationParameters:
        store = self.store.clone(detach = True, prefixes = self._store_prefixes(clone_opts))
        if exists(indices):
            store.keep_(indices, self.pop_size)

        pop_data = None
        if exists(self._pop_leaf):
            pop_data = slice_pop(self._pop_leaf, indices) if exists(indices) else self._pop_leaf.detach().clone()
        shared_data = self._shared_leaf.detach().clone() if exists(self._shared_leaf) else None

        return self._spawn(
            store, pop_data, shared_data,
            pop_size = len(indices) if exists(indices) else self.pop_size,
            template = deepcopy(self._template),
            functional = False,
            clone_opts = clone_opts,
        )

    def clone(self, *, include_opt: bool = True) -> PopulationParameters:
        self._require_pop('clone()')
        if not self.is_functional:
            return self.sliced(clone_opts = include_opt)

        pop_leaf = self._pop_leaf.clone() if exists(self._pop_leaf) else None
        shared_leaf = self._shared_leaf.clone() if exists(self._shared_leaf) else None
        return self._spawn(
            self.store.clone(detach = False, prefixes = self._store_prefixes(include_opt)),
            pop_leaf, shared_leaf,
            functional = True,
            clone_opts = include_opt,
        )

    def detach(self, *, include_opt: bool = True) -> PopulationParameters:
        self._require_pop('detach()')
        return self.sliced(clone_opts = include_opt)

    def _apply(self, fn, recurse = True):
        res = super()._apply(fn, recurse = recurse)
        if exists(self._func_leaves):
            self._func_leaves = tuple(fn(t) if exists(t) else None for t in self._func_leaves)
        return res

    # accessors

    @property
    def is_population(self) -> bool:
        return exists(self.pop_size)

    @property
    def is_functional(self) -> bool:
        return exists(self._func_leaves)

    @property
    def template(self) -> Module:
        return self._template if self.is_population else self._noop_net

    @property
    def trainable_specs(self) -> list[TensorSpec]:
        return self._schema.trainable

    @property
    def _pop_leaf(self) -> Tensor | None:
        return self._func_leaves[0] if exists(self._func_leaves) else self._pop_param

    @property
    def _shared_leaf(self) -> Tensor | None:
        return self._func_leaves[1] if exists(self._func_leaves) else self._shared_param

    @property
    def _leaves(self) -> list[Tensor]:
        return [t for t in (self._pop_leaf, self._shared_leaf) if exists(t)]

    @property
    def buffer_names(self) -> list[str]:
        entries = self.store.items('buffer/') if self.is_population else self._noop_net.named_buffers()
        return [name for name, _ in entries]

    @property
    def flat_params(self) -> Tensor:
        if not self.is_population:
            trainable = [p for p in self._noop_net.parameters() if p.requires_grad]
            if len({p.dtype for p in trainable}) > 1:
                raise ValueError('flat_params requires a single trainable parameter dtype')
            return pack(trainable, '*')[0] if trainable else torch.empty(0)
        return self._pop_leaf if exists(self._pop_leaf) else torch.empty((self.pop_size, 0))

    @property
    def flat_shared_params(self) -> Tensor | None:
        return self._shared_leaf

    def _require_pop(self, op: str):
        if not self.is_population:
            raise ValueError(f'{op} requires a populated PopulationParameters')

    def has_param(self, name: str) -> bool:
        return name in self._schema.by_name

    def is_shared_param(self, name: str) -> bool:
        return self._schema.by_name[name].shared

    def register_opt_store(self, name: str) -> OptStore:
        return self.opt_stores.setdefault(name, OptStore(self, name))

    def _opt_key(self, opt = None) -> str | None:
        return self._default_opt_name if not exists(opt) else (opt if isinstance(opt, str) else opt.name)

    def _opt_owner(self, opt = None):
        key = self._opt_key(opt)
        return self._optimizers.get(key) if exists(key) else None

    def opt_slot_flat(self, slot: str, opt = None) -> Tensor | None:
        store = self.opt_stores.get(self._opt_key(opt))
        return store.get(f'slot/{slot}/pop') if exists(store) else None

    def opt_aux(self, name: str, param_name: str, opt = None) -> Tensor | None:
        store = self.opt_stores.get(self._opt_key(opt))
        return store.get(f'aux/{name}/{param_name}') if exists(store) else None

    def unpack_flat(self, flat: Tensor) -> dict[str, Tensor]:
        specs = self._schema.pop_trainable if self.is_population else self._schema.trainable
        return dict(zip((s.name for s in specs), unpack_tensors(flat, [s.shape for s in specs])))

    # views & gradients

    def _param_view(self, spec: TensorSpec) -> Tensor:
        if not spec.requires_grad:
            return self.store[f'frozen/{spec.name}']
        return slice_spec(spec, self._shared_leaf if spec.shared else self._pop_leaf)

    def _member_tensors(self, idx: int, *, clone: bool = False, track_grads: bool = False):
        if not self.is_population:
            params, buffers = self.named_parameters_dict(), self.named_buffers_dict()
        else:
            named = self.named_parameters_dict(track_grads = track_grads)
            params = {s.name: (named[s.name] if s.shared else named[s.name][idx]) for s in self._schema.specs}
            buffers = {name: buf[idx] for name, buf in self.store.items('buffer/')}

        if not clone:
            return params, buffers
        return (
            {k: t.detach().clone() for k, t in params.items()},
            {k: t.detach().clone() for k, t in buffers.items()},
        )

    def _copy_grad_(self, spec: TensorSpec, dst_param: Tensor, grad: Tensor, idx: int):
        if not self.is_population:
            dst_param.grad = grad.detach().clone()
            return
        leaf = self._shared_leaf if spec.shared else self._pop_leaf
        if not exists(leaf.grad):
            leaf.grad = torch.zeros_like(leaf)
        g_dst = slice_spec(spec, leaf.grad)
        if not spec.shared:
            g_dst = g_dst[idx]
        g_dst.copy_(grad.detach().to(g_dst))
        if exists(self._active_grads):
            self._active_grads.add(spec.name)

    def _copy_member_state_(
        self,
        idx: int,
        src_params: Mapping[str, Tensor] | None,
        src_buffers: Mapping[str, Tensor] | None,
        *,
        check_shapes: bool = False,
        skip_shared: bool = False,
        sync_grads: bool = False,
    ):
        if exists(src_params):
            for name, dst in self.named_parameters_dict().items():
                spec = self._schema.by_name[name]
                if (skip_shared and spec.shared) or name not in src_params:
                    continue
                src = src_params[name]
                target = dst[idx] if (self.is_population and not spec.shared) else dst
                copy_tensor_(target, src, 'parameter', name, check_shapes = check_shapes)
                if sync_grads and spec.requires_grad and exists(src.grad):
                    self._copy_grad_(spec, dst, src.grad, idx)

        if exists(src_buffers):
            for name, dst in self.named_buffers_dict().items():
                if name in src_buffers:
                    target = dst[idx] if self.is_population else dst
                    copy_tensor_(target, src_buffers[name], 'buffer', name, check_shapes = check_shapes)

    def named_parameters_dict(self, *, track_grads: bool = False) -> dict[str, Tensor]:
        if not self.is_population:
            return dict(self._noop_net.named_parameters())

        should_hook = track_grads and torch.is_grad_enabled() and not self.is_functional
        if should_hook and not exists(self._active_grads):
            self._active_grads = set()

        named = {}
        for spec in self._schema.specs:
            view = self._param_view(spec)
            if should_hook and spec.requires_grad and view.requires_grad:
                view.register_hook(
                    lambda _g, name = spec.name: self._active_grads.add(name) if exists(self._active_grads) else None
                )
            named[spec.name] = view
        return named

    def named_grads_dict(self) -> dict[str, Tensor | None]:
        if not self.is_population:
            return {name: p.grad for name, p in self._noop_net.named_parameters()}

        grads = {}
        for spec in self._schema.specs:
            leaf = (self._shared_leaf if spec.shared else self._pop_leaf) if spec.requires_grad else None
            grad = leaf.grad if exists(leaf) else None
            inactive = exists(self._active_grads) and spec.name not in self._active_grads
            grads[spec.name] = None if (not exists(grad) or inactive) else slice_spec(spec, grad)
        return grads

    def named_buffers_dict(self) -> dict[str, Tensor]:
        return self.store.as_dict('buffer/') if self.is_population else dict(self._noop_net.named_buffers())

    def named_optimizer_state(self, opt = None) -> dict[str, dict[str, Tensor]] | None:
        owner = self._opt_owner(opt)
        return owner.named_state(self) if exists(owner) else None

    def autograd(
        self,
        loss: Tensor,
        *,
        names: Iterable[str] | None = None,
        retain_graph: bool | None = None,
        create_graph: bool = False,
    ) -> dict[str, Tensor]:
        trainable = self._schema.trainable
        active = set(names) if exists(names) else {s.name for s in trainable}
        grad_kw = dict(retain_graph = default(retain_graph, create_graph), create_graph = create_graph, allow_unused = True)

        if not self.is_population:
            named = self.named_parameters_dict()
            keys = [s.name for s in trainable if s.name in active]
            raw = dict(zip(keys, torch_grad(loss, [named[k] for k in keys], **grad_kw))) if keys else {}
            return {s.name: default(raw.get(s.name), torch.zeros_like(named[s.name])) for s in trainable}

        need_pop = exists(self._pop_leaf) and any(s.name in active and not s.shared for s in trainable)
        need_shared = exists(self._shared_leaf) and any(s.name in active and s.shared for s in trainable)
        targets = ([self._pop_leaf] if need_pop else []) + ([self._shared_leaf] if need_shared else [])
        raw = torch_grad(loss, targets, **grad_kw) if targets else ()
        pop_grad = raw[0] if need_pop else None
        shared_grad = raw[-1] if need_shared else None

        out = {}
        for spec in trainable:
            grad = (shared_grad if spec.shared else pop_grad) if spec.name in active else None
            out[spec.name] = slice_spec(spec, grad) if exists(grad) else torch.zeros_like(self._param_view(spec))
        return out

    def zero_grad_(self, members = None, set_to_none: bool = True):
        indices = self.resolve_members(members) if exists(members) else None
        if exists(indices) and not indices:
            return

        if exists(indices) and self.is_population:
            with torch.no_grad():
                if exists(self._pop_leaf) and exists(self._pop_leaf.grad):
                    self._pop_leaf.grad[indices] = 0.
            return

        for param in (self._leaves if self.is_population else self._noop_net.parameters()):
            if set_to_none:
                param.grad = None
            elif exists(param.grad):
                param.grad.zero_()
        self._active_grads = None

    # indexing & PyTorch-compatible dicts

    def _bundle(self, named_params: dict[str, Tensor], named_buffers: dict[str, Tensor], *, include_state_dict: bool = False, **extra) -> dict:
        out = {
            'parameters': list(named_params.values()),
            'named_parameters': named_params,
            'buffers': list(named_buffers.values()),
            'named_buffers': named_buffers,
            **extra,
        }
        if include_state_dict:
            out['state_dict'] = {
                sd_key: (named_params if is_param else named_buffers)[canonical]
                for sd_key, canonical, is_param in self._schema.state_dict_map
                if canonical in (named_params if is_param else named_buffers)
            }
        return out

    def _opt_bundle(self, opt = None, *, idx: int | None = None, clone: bool = False) -> dict:
        owner = self._opt_owner(opt)
        if not exists(owner):
            return {'optimizer_state': None, 'named_optimizer_state': None}
        return {
            'optimizer_state': owner.state_dict_for(self, idx = idx, clone = clone),
            'named_optimizer_state': owner.named_state(self, idx = idx, clone = clone),
        }

    def as_dict(self, opt = None) -> dict:
        return self._bundle(self.named_parameters_dict(), self.named_buffers_dict(), **self._opt_bundle(opt))

    def _normalize_index(self, index) -> int:
        return normalize_index(index, len(self))

    def resolve_members(self, selector, *, non_empty: bool = False) -> list[int]:
        return resolve_indices(selector, len(self), non_empty = non_empty)

    def individual(self, index: int = 0, *, opt = None, clone: bool = False) -> dict:
        idx = self._normalize_index(index)
        params, buffers = self._member_tensors(idx, clone = clone)
        opt_idx = idx if self.is_population else None
        return self._bundle(params, buffers, include_state_dict = True, **self._opt_bundle(opt, idx = opt_idx, clone = clone))

    @torch.no_grad()
    def load_individual_dict_(self, index: int, data: Mapping, *, opt = None, sync_grads: bool = False) -> PopulationParameters:
        idx = self._normalize_index(index)
        sd = data.get('state_dict', {})
        src_params = data.get('named_parameters') or {s.name: sd[s.name] for s in self._schema.specs if s.name in sd}
        src_buffers = data.get('named_buffers') or {name: sd[name] for name in self.buffer_names if name in sd}
        self._copy_member_state_(idx, src_params, src_buffers, check_shapes = True, sync_grads = sync_grads)

        if exists(data.get('named_optimizer_state')) or exists(data.get('optimizer_state')):
            owner = self._opt_owner(opt)
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
            raise TypeError(f'item assignment expects a dict/Mapping, got {type(value).__name__}')
        self.load_individual_dict_(index, value)

    # lifecycle

    def mean_dict(self, weights: Tensor | Sequence[float] | None = None) -> dict:
        if not self.is_population:
            return self.individual(0, clone = True)

        weight_vec = None
        if exists(weights):
            weight_vec = weights.detach().float() if is_tensor(weights) else torch.tensor([float(x) for x in weights])
            if weight_vec.shape != (self.pop_size,):
                raise ValueError(f'weights must have shape ({self.pop_size},), got {tuple(weight_vec.shape)}')

        def average(t: Tensor) -> Tensor:
            t = t.detach()
            if not t.is_floating_point():
                return t[0].clone()
            return torch.tensordot(weight_vec.to(t), t, dims = 1) if exists(weight_vec) else t.mean(dim = 0)

        named_params = {
            s.name: (self._param_view(s).detach().clone() if s.shared else average(self._param_view(s)))
            for s in self._schema.specs
        }
        named_buffers = {name: average(buf) for name, buf in self.store.items('buffer/')}
        return self._bundle(named_params, named_buffers, include_state_dict = True)

    def mean(self, weights: Tensor | Sequence[float] | None = None, *, training: bool | None = None) -> Module:
        if not self.is_population:
            cloned = deepcopy(self._noop_net)
            if exists(training):
                cloned.train(training)
            return cloned

        avg = self.mean_dict(weights = weights)
        return clone_template(self._template, avg['named_parameters'], avg['named_buffers'], training = training)

    @torch.no_grad()
    def lerp_(self, other: PopulationParameters, weight: float) -> PopulationParameters:
        param_pairs = (
            zip(self._leaves, other._leaves)
            if self.is_population
            else zip(self.named_parameters_dict().values(), other.named_parameters_dict().values())
        )
        other_buffers = other.named_buffers_dict()
        pairs = [
            *param_pairs,
            *((dst, other_buffers[k]) for k, dst in self.named_buffers_dict().items() if k in other_buffers),
            *((dst, other.store[f'frozen/{k}']) for k, dst in self.store.items('frozen/') if f'frozen/{k}' in other.store),
        ]
        for dst, src in pairs:
            if dst.is_floating_point():
                dst.lerp_(src.detach().to(dst), weight)
        return self

    @torch.no_grad()
    def copy_from_(self, other: PopulationParameters, *, members = None, include_opt: bool = True) -> PopulationParameters:
        if not self.is_population:
            self._copy_member_state_(0, other.named_parameters_dict(), other.named_buffers_dict())
            if include_opt:
                self.store.copy_from_(other.store, prefixes = ('opt/',))
            return self

        indices = self.resolve_members(members) if exists(members) else None
        if exists(indices) and not indices:
            return self

        if exists(self._pop_leaf) and exists(other._pop_leaf):
            if exists(indices):
                self._pop_leaf[indices] = other._pop_leaf[indices].detach().to(self._pop_leaf)
            else:
                self._pop_leaf.copy_(other._pop_leaf.detach().to(self._pop_leaf))

        if not exists(indices) and exists(self._shared_leaf) and exists(other._shared_leaf):
            self._shared_leaf.copy_(other._shared_leaf.detach().to(self._shared_leaf))

        self.store.copy_from_(other.store, prefixes = self._store_prefixes(include_opt), indices = indices, pop_size = self.pop_size)
        return self

    def _reset_opts_(self, indices: list[int] | None):
        for store in self.opt_stores.values():
            store.reset_members_(indices)

    @torch.no_grad()
    def reset_(
        self,
        selector = None,
        *,
        init: Literal['reinit'] | Module | PopulationParameters | Callable[[Module], None] = 'reinit',
        reset_opt: bool = True,
        reset_buffers: bool = True,
    ) -> PopulationParameters:
        ref_pp = init if isinstance(init, PopulationParameters) else getattr(init, 'population_parameters', None)
        has_ref_pp = isinstance(ref_pp, PopulationParameters)
        indices = [0] if not self.is_population else (list(range(self.pop_size)) if not exists(selector) else self.resolve_members(selector))
        if not indices:
            return self

        if has_ref_pp and ref_pp._schema.param_names != self._schema.param_names:
            raise ValueError('reference population parameter names do not match target population')

        if self.is_population and has_ref_pp and ref_pp.is_population:
            if ref_pp.pop_size != self.pop_size:
                raise ValueError(f'reference population size ({ref_pp.pop_size}) does not match target ({self.pop_size})')
            if exists(self._pop_leaf) and exists(ref_pp._pop_leaf):
                self._pop_leaf[indices] = ref_pp._pop_leaf[indices].detach().to(self._pop_leaf)
            self.store.copy_from_(ref_pp.store, prefixes = self._store_prefixes(False, reset_buffers), indices = indices, pop_size = self.pop_size)
        elif has_ref_pp or isinstance(init, Module):
            src_p, src_b = ref_pp._member_tensors(0) if has_ref_pp else (dict(init.named_parameters()), dict(init.named_buffers()))
            src_b = src_b if reset_buffers else None
            for i in indices:
                self._copy_member_state_(i, src_p, src_b, skip_shared = self.is_population)
        else:
            for i in indices:
                if init == 'reinit' and exists(self._factory):
                    fresh = self._factory()
                elif not self.is_population:
                    saved = {k: b.detach().clone() for k, b in self._noop_net.named_buffers()} if not reset_buffers else None
                    init_module(self._noop_net, init)
                    if exists(saved):
                        for name, buf in self._noop_net.named_buffers():
                            buf.copy_(saved[name])
                    continue
                else:
                    fresh = clone_template(self._template, *self._member_tensors(i), training = True)
                    init_module(fresh, init)
                src_b = dict(fresh.named_buffers()) if reset_buffers else None
                self._copy_member_state_(i, dict(fresh.named_parameters()), src_b, skip_shared = self.is_population)

        if reset_opt:
            self._reset_opts_(indices if (self.is_population and exists(selector)) else None)
        return self

    def _extract_source(self, source):
        if is_int_index(source):
            return self._member_tensors(self._normalize_index(source), clone = True)
        src_pp = source if isinstance(source, PopulationParameters) else getattr(source, 'population_parameters', None)
        if isinstance(src_pp, PopulationParameters):
            if src_pp.is_population and src_pp.pop_size != 1:
                raise ValueError(f'cannot extract a single member from population of size {src_pp.pop_size}')
            return src_pp._member_tensors(0, clone = True)
        if hasattr(source, '_pop') and getattr(source, 'is_single', False):
            return source._pop.population_parameters._member_tensors(source._index, clone = True)
        if isinstance(source, Module):
            return dict(source.named_parameters()), dict(source.named_buffers())
        if isinstance(source, Mapping):
            if 'named_parameters' in source:
                return dict(source['named_parameters']), source.get('named_buffers')
            return dict(source), None
        if hasattr(source, 'items'):
            return dict(source.items()), None
        raise TypeError(f'source must be member index, nn.Module, or param mapping, got {type(source).__name__}')

    @torch.no_grad()
    def replace_(self, index: int, source, *, reset_opt: bool = True) -> PopulationParameters:
        self._require_pop('replace_()')
        idx = self._normalize_index(index)
        self._copy_member_state_(idx, *self._extract_source(source), skip_shared = True)
        if reset_opt:
            self._reset_opts_([idx])
        return self

    def _finish_resize(self, new_size: int):
        self.pop_size = new_size
        self._active_grads = None
        for opt in self._optimizers.values():
            if opt._pp is self:
                opt._sync_optimizer_state()

    @torch.no_grad()
    def keep_(self, members) -> PopulationParameters:
        self._require_pop('keep_()')
        if self.is_functional:
            raise ValueError('keep_() requires a detached/backed PopulationParameters; call detach() first')

        keep = list(dict.fromkeys(self.resolve_members(members, non_empty = True)))
        old_pop = self.pop_size
        if exists(self._pop_param):
            old = self._pop_param
            sliced = Parameter(slice_pop(old, keep), requires_grad = True)
            if exists(old.grad):
                sliced.grad = slice_pop(old.grad, keep)
            self._pop_param = sliced
        self.store.keep_(keep, old_pop)
        self._finish_resize(len(keep))
        return self

    def delete_(self, members) -> PopulationParameters:
        self._require_pop('delete_()')
        drop = set(self.resolve_members(members))
        return self.keep_([i for i in range(self.pop_size) if i not in drop])

    @torch.no_grad()
    def append_(self, source) -> int:
        self._require_pop('append_()')
        if self.is_functional:
            raise ValueError('append_() requires a detached/backed PopulationParameters; call detach() first')

        src_params, src_buffers = self._extract_source(source)
        old_pop = self.pop_size
        if exists(self._pop_param):
            self._pop_param = Parameter(pad_pop(self._pop_param), requires_grad = True)
        self.store.append_(old_pop)
        self._finish_resize(old_pop + 1)
        self._copy_member_state_(old_pop, src_params, src_buffers, skip_shared = True)
        return old_pop

    def extra_repr(self) -> str:
        num_params = sum(s.numel for s in self._schema.trainable)
        if not self.is_population:
            return f'noop, num_params={num_params}'
        mode = 'functional' if self.is_functional else 'backed'
        return f'pop_size={self.pop_size}, num_params={num_params}, mode={mode}'
