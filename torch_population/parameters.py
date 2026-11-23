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

def _validate_maps(kind: str, maps: list[dict[str, Tensor]]):
    ref, names = maps[0], list(maps[0])
    for i, cur in enumerate(maps[1:], start = 1):
        if list(cur) != names:
            raise ValueError(f'module {i} {kind} names do not match module 0')
        for k, t in ref.items():
            if cur[k].shape != t.shape:
                raise ValueError(f'{kind} {k!r} shape mismatch: {t.shape} vs {cur[k].shape}')

def _pack_group(specs: list[TensorSpec], mod_params: list[dict[str, Tensor]], is_shared: bool) -> Parameter | None:
    if not specs:
        return None
    if len({mod_params[0][s.name].dtype for s in specs}) > 1:
        raise ValueError('trainable parameters in a Population must share the same dtype')
    stacked = [torch.stack([mp[s.name].detach() for mp in mod_params]) for s in specs]
    if is_shared:
        packed = pack([t.mean(0) if t.is_floating_point() else t[0] for t in stacked], '*')[0]
    else:
        packed = pack(stacked, 'p *')[0]
    return Parameter(packed, requires_grad = True)

def _copy_tensor_(dst: Tensor, src: Tensor, kind: str, name: str, check_shapes: bool):
    if check_shapes and src.shape != dst.shape:
        raise ValueError(f'{kind} {name!r} shape mismatch: {tuple(dst.shape)} vs {tuple(src.shape)}')
    dst.copy_(src.detach().to(dst))

# parameter layout schema

class _Schema:
    def __init__(self, ref: Module | None = None, shared = ()):
        self.specs: list[TensorSpec] = []
        self.state_dict_map: list[tuple[str, str, bool]] = []

        if exists(ref):
            named = list(ref.named_parameters())
            pop_off = sh_off = tidx = 0
            for name, p in named:
                sh = matches_pattern(name, shared)
                if not p.requires_grad:
                    self.specs.append(TensorSpec(name, tuple(p.shape), p.numel(), 0, False, None, sh))
                    continue
                off = sh_off if sh else pop_off
                self.specs.append(TensorSpec(name, tuple(p.shape), p.numel(), off, True, tidx, sh))
                if sh:
                    sh_off += p.numel()
                else:
                    pop_off += p.numel()
                tidx += 1

            by_id = {id(p): (n, True) for n, p in named}
            by_id.update({id(b): (n, False) for n, b in ref.named_buffers()})
            pnames = {s.name for s in self.specs}
            self.state_dict_map = [(k, *by_id.get(id(t), (k, k in pnames))) for k, t in ref.state_dict().items()]

        self.by_name = {s.name: s for s in self.specs}
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
        self._schema = _Schema()
        self._func_leaves: tuple[Tensor | None, Tensor | None] | None = None

    # init & cloning

    def init_noop(self, net: Module):
        self.pop_size = None
        object.__setattr__(self, '_noop_net', net)
        object.__setattr__(self, '_template', None)
        self._schema = _Schema(net)
        self._func_leaves = None
        self._active_grads = None

    def init_from_modules(self, modules: list[Module], template: Module, *, shared = ()):
        ref = modules[0]
        mod_p = [dict(m.named_parameters()) for m in modules]
        mod_b = [dict(m.named_buffers()) for m in modules]
        _validate_maps('parameter', mod_p)
        _validate_maps('buffer', mod_b)

        schema = _Schema(ref, shared)
        self.pop_size = len(modules)
        object.__setattr__(self, '_noop_net', None)
        object.__setattr__(self, '_template', template)
        self._schema = schema
        self._func_leaves = None
        self._active_grads = None

        self._pop_param = _pack_group(schema.pop_trainable, mod_p, is_shared = False)
        self._shared_param = _pack_group(schema.shared_trainable, mod_p, is_shared = True)

        self.store.clear_('frozen/')
        for s in schema.specs:
            if s.requires_grad:
                continue
            val = torch.stack([mp[s.name].detach().clone() for mp in mod_p])
            if s.shared:
                val = val.mean(0) if val.is_floating_point() else val[0]
            self.store.set(f'frozen/{s.name}', val, populated = not s.shared)

        sd_keys = set(ref.state_dict())
        self.store.clear_('buffer/')
        for name in mod_b[0]:
            stacked = torch.stack([mb[name].detach().clone() for mb in mod_b])
            self.store.set(f'buffer/{name}', stacked, populated = True, persistent = name in sd_keys, append_repeat = True)

    def _spawn(self, store: TensorStore, pop_flat: Tensor | None, shared_flat: Tensor | None, *, pop_size = None, template = None, functional: bool = False, clone_opts: bool = True) -> PopulationParameters:
        out = PopulationParameters()
        out.pop_size = default(pop_size, self.pop_size)
        object.__setattr__(out, '_noop_net', self._noop_net)
        object.__setattr__(out, '_template', default(template, self._template))
        out._schema = self._schema
        out._func_leaves = (pop_flat, shared_flat) if functional else None
        if not functional:
            out._pop_param = Parameter(pop_flat, requires_grad = True) if exists(pop_flat) else None
            out._shared_param = Parameter(shared_flat, requires_grad = True) if exists(shared_flat) else None
        out.store = store
        if clone_opts:
            out.opt_stores = {k: v.clone_handle(out) for k, v in self.opt_stores.items()}
            out._optimizers = dict(self._optimizers)
            out._default_opt_name = self._default_opt_name
        return out

    def sliced(self, indices: list[int] | None = None, *, clone_opts: bool = False) -> PopulationParameters:
        prefixes = ('frozen/', 'buffer/', 'opt/') if clone_opts else ('frozen/', 'buffer/')
        store = self.store.clone(detach = True, prefixes = prefixes)
        if exists(indices):
            store.keep_(indices, self.pop_size)
        pop_data = None
        if exists(self._pop_param):
            pop_data = slice_pop(self._pop_param, indices) if exists(indices) else self._pop_param.detach().clone()
        sh_data = self._shared_param.detach().clone() if exists(self._shared_param) else None
        return self._spawn(store, pop_data, sh_data, pop_size = len(indices) if exists(indices) else self.pop_size, template = deepcopy(self._template), clone_opts = clone_opts)

    def _spawn_functional(self, pop_flat: Tensor | None, shared_flat: Tensor | None) -> PopulationParameters:
        store = self.store.clone(detach = True, prefixes = ('frozen/', 'buffer/', 'opt/'), shallow_prefixes = ('opt/',))
        return self._spawn(store, pop_flat, shared_flat, functional = True)

    def clone(self, *, _detach: bool = False) -> PopulationParameters:
        self._require_pop('detach()' if _detach else 'clone()')
        if not self.is_functional:
            return self.sliced(clone_opts = True)
        pop_t, sh_t = (
            (t.detach().clone().requires_grad_(True) if _detach else t.clone()) if exists(t) else None
            for t in self._func_leaves
        )
        return self._spawn(self.store.clone(detach = _detach), pop_t, sh_t, functional = True)

    def detach(self) -> PopulationParameters:
        return self.clone(_detach = True)

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
        return [k for k, _ in entries]

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

    # views, autograd & grad zeroing

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
            buffers = {k: t[idx] for k, t in self.store.items('buffer/')}
        if not clone:
            return params, buffers
        return {k: t.detach().clone() for k, t in params.items()}, {k: t.detach().clone() for k, t in buffers.items()}

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

    def _copy_member_state_(self, idx: int, src_p: Mapping[str, Tensor] | None, src_b: Mapping[str, Tensor] | None, *, check_shapes: bool = False, skip_shared: bool = False, sync_grads: bool = False):
        if exists(src_p):
            for name, dst in self.named_parameters_dict().items():
                spec = self._schema.by_name[name]
                if (skip_shared and spec.shared) or name not in src_p:
                    continue
                src = src_p[name]
                target = dst[idx] if (self.is_population and not spec.shared) else dst
                _copy_tensor_(target, src, 'parameter', name, check_shapes)
                if sync_grads and spec.requires_grad and exists(src.grad):
                    self._copy_grad_(spec, dst, src.grad, idx)

        if exists(src_b):
            for name, dst in self.named_buffers_dict().items():
                if name in src_b:
                    target = dst[idx] if self.is_population else dst
                    _copy_tensor_(target, src_b[name], 'buffer', name, check_shapes)

    def named_parameters_dict(self, *, track_grads: bool = False) -> dict[str, Tensor]:
        if not self.is_population:
            return dict(self._noop_net.named_parameters())

        # track which parameter slices actually receive gradients during backward
        hook = track_grads and torch.is_grad_enabled() and not self.is_functional
        if hook and not exists(self._active_grads):
            self._active_grads = set()

        named = {}
        for s in self._schema.specs:
            v = self._param_view(s)
            if hook and s.requires_grad and v.requires_grad:
                v.register_hook(lambda _g, n = s.name: self._active_grads.add(n) if exists(self._active_grads) else None)
            named[s.name] = v
        return named

    def named_grads_dict(self) -> dict[str, Tensor | None]:
        if not self.is_population:
            return {n: p.grad for n, p in self._noop_net.named_parameters()}
        out = {}
        for s in self._schema.specs:
            leaf = (self._shared_leaf if s.shared else self._pop_leaf) if s.requires_grad else None
            g = leaf.grad if exists(leaf) else None
            inactive = exists(self._active_grads) and s.name not in self._active_grads
            out[s.name] = None if (not exists(g) or inactive) else slice_spec(s, g)
        return out

    def named_buffers_dict(self) -> dict[str, Tensor]:
        return self.store.as_dict('buffer/') if self.is_population else dict(self._noop_net.named_buffers())

    def named_optimizer_state(self, opt = None) -> dict[str, dict[str, Tensor]] | None:
        owner = self._opt_owner(opt)
        return owner.named_state(self) if exists(owner) else None

    def autograd(self, loss: Tensor, *, names: Iterable[str] | None = None, retain_graph: bool | None = None, create_graph: bool = False) -> dict[str, Tensor]:
        trainable = self._schema.trainable
        active = set(names) if exists(names) else {s.name for s in trainable}
        kw = {'retain_graph': default(retain_graph, create_graph), 'create_graph': create_graph, 'allow_unused': True}

        if not self.is_population:
            named = self.named_parameters_dict()
            keys = [s.name for s in trainable if s.name in active]
            raw = dict(zip(keys, torch_grad(loss, [named[k] for k in keys], **kw))) if keys else {}
            return {s.name: default(raw.get(s.name), torch.zeros_like(named[s.name])) for s in trainable}

        need_pop = any(s.name in active and not s.shared for s in trainable) and exists(self._pop_leaf)
        need_sh = any(s.name in active and s.shared for s in trainable) and exists(self._shared_leaf)
        targets = ([self._pop_leaf] if need_pop else []) + ([self._shared_leaf] if need_sh else [])
        raw = torch_grad(loss, targets, **kw) if targets else ()
        pop_g = raw[0] if need_pop else None
        sh_g = raw[-1] if need_sh else None

        out = {}
        for s in trainable:
            g = (sh_g if s.shared else pop_g) if s.name in active else None
            out[s.name] = slice_spec(s, g) if exists(g) else torch.zeros_like(self._param_view(s))
        return out

    def zero_grad_(self, members = None, set_to_none: bool = True):
        idx = self.resolve_members(members) if exists(members) else None
        if exists(idx) and not idx:
            return
        if exists(idx) and self.is_population:
            with torch.no_grad():
                if exists(self._pop_leaf) and exists(self._pop_leaf.grad):
                    self._pop_leaf.grad[idx] = 0.
            return

        for p in (self._leaves if self.is_population else self._noop_net.parameters()):
            if set_to_none:
                p.grad = None
            elif exists(p.grad):
                p.grad.zero_()
        self._active_grads = None

    # indexing & PyTorch-compatible dicts

    def _bundle(self, named_p, named_b, *, state_dict = False, **extra) -> dict:
        out = {
            'parameters': list(named_p.values()), 'named_parameters': named_p,
            'buffers': list(named_b.values()), 'named_buffers': named_b, **extra,
        }
        if state_dict:
            out['state_dict'] = {
                sd_k: (named_p if is_p else named_b)[canon]
                for sd_k, canon, is_p in self._schema.state_dict_map
                if canon in (named_p if is_p else named_b)
            }
        return out

    def as_dict(self, opt = None) -> dict:
        owner = self._opt_owner(opt)
        return self._bundle(
            self.named_parameters_dict(), self.named_buffers_dict(),
            optimizer_state = owner.state_dict_for(self) if exists(owner) else None,
            named_optimizer_state = owner.named_state(self) if exists(owner) else None,
        )

    def _normalize_index(self, index) -> int:
        return normalize_index(index, len(self))

    def resolve_members(self, selector, *, non_empty: bool = False) -> list[int]:
        return resolve_indices(selector, len(self), non_empty = non_empty)

    def individual(self, index: int = 0, *, opt = None, clone: bool = False) -> dict:
        idx = self._normalize_index(index)
        params, buffers = self._member_tensors(idx, clone = clone)
        owner = self._opt_owner(opt)
        opt_idx = idx if self.is_population else None
        return self._bundle(
            params, buffers, state_dict = True,
            optimizer_state = owner.state_dict_for(self, idx = opt_idx, clone = clone) if exists(owner) else None,
            named_optimizer_state = owner.named_state(self, idx = opt_idx, clone = clone) if exists(owner) else None,
        )

    @torch.no_grad()
    def load_individual_dict_(self, index: int, data: Mapping, *, opt = None, sync_grads: bool = False) -> PopulationParameters:
        idx = self._normalize_index(index)
        sd = data.get('state_dict', {})
        src_p = data.get('named_parameters') or {s.name: sd[s.name] for s in self._schema.specs if s.name in sd}
        src_b = data.get('named_buffers') or {k: sd[k] for k in self.buffer_names if k in sd}
        self._copy_member_state_(idx, src_p, src_b, check_shapes = True, sync_grads = sync_grads)

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

    # lifecycle ops

    def mean_dict(self, weights: Tensor | Sequence[float] | None = None) -> dict:
        if not self.is_population:
            return self.individual(0, clone = True)

        w = None
        if exists(weights):
            w = weights.detach().float() if is_tensor(weights) else torch.tensor([float(x) for x in weights])
            if w.shape != (self.pop_size,):
                raise ValueError(f'weights must have shape ({self.pop_size},), got {tuple(w.shape)}')

        def reduce_t(t: Tensor) -> Tensor:
            t = t.detach()
            if not t.is_floating_point():
                return t[0].clone()
            return torch.tensordot(w.to(t), t, dims = 1) if exists(w) else t.mean(0)

        named_p = {s.name: (self._param_view(s).detach().clone() if s.shared else reduce_t(self._param_view(s))) for s in self._schema.specs}
        named_b = {k: reduce_t(b) for k, b in self.store.items('buffer/')}
        return self._bundle(named_p, named_b, state_dict = True)

    @torch.no_grad()
    def lerp_(self, other: PopulationParameters, weight: float) -> PopulationParameters:
        for dst, src in zip(self._leaves, other._leaves):
            if dst.is_floating_point():
                dst.lerp_(src.detach().to(dst), weight)
        for k in self.store:
            if not k.startswith(('frozen/', 'buffer/')):
                continue
            src = other.store.get(k)
            dst = self.store[k]
            if exists(src) and dst.is_floating_point():
                dst.lerp_(src.detach().to(dst), weight)
        return self

    @torch.no_grad()
    def copy_from_(self, other: PopulationParameters, *, members = None, include_opt: bool = True) -> PopulationParameters:
        idx = self.resolve_members(members) if (exists(members) and self.is_population) else None
        if exists(idx) and not idx:
            return self
        if exists(self._pop_leaf) and exists(other._pop_leaf):
            cast = other._pop_leaf.detach().to(self._pop_leaf)
            if exists(idx):
                self._pop_leaf[idx] = cast[idx]
            else:
                self._pop_leaf.copy_(cast)
        if not exists(idx) and exists(self._shared_leaf) and exists(other._shared_leaf):
            self._shared_leaf.copy_(other._shared_leaf.detach().to(self._shared_leaf))
        prefixes = ('frozen/', 'buffer/', 'opt/') if include_opt else ('frozen/', 'buffer/')
        self.store.copy_from_(other.store, prefixes = prefixes, indices = idx, pop_size = self.pop_size)
        return self

    def _reset_opts_(self, indices: list[int] | None):
        for store in self.opt_stores.values():
            store.reset_members_(indices)

    @torch.no_grad()
    def reset_(self, selector = None, *, init: Literal['reinit'] | Callable[[Module], None] = 'reinit', reset_opt: bool = True, reset_buffers: bool = True) -> PopulationParameters:
        if not self.is_population:
            init_module(self._noop_net, init)
            if reset_opt:
                self._reset_opts_(None)
            return self

        idx = list(range(self.pop_size)) if not exists(selector) else self.resolve_members(selector)
        if not idx:
            return self

        for i in idx:
            fresh = clone_template(self._template, *self._member_tensors(i), training = True)
            init_module(fresh, init)
            fresh_b = dict(fresh.named_buffers()) if reset_buffers else None
            self._copy_member_state_(i, dict(fresh.named_parameters()), fresh_b, skip_shared = True)

        if reset_opt:
            self._reset_opts_(idx)
        return self

    def _extract_source(self, source):
        if is_int_index(source):
            return self._member_tensors(self._normalize_index(source), clone = True)
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
        src_p, src_b = self._extract_source(source)
        old_pop = self.pop_size
        if exists(self._pop_param):
            self._pop_param = Parameter(pad_pop(self._pop_param), requires_grad = True)
        self.store.append_(old_pop)
        self._finish_resize(old_pop + 1)
        self._copy_member_state_(old_pop, src_p, src_b, skip_shared = True)
        return old_pop

    def extra_repr(self) -> str:
        n = sum(s.numel for s in self._schema.trainable)
        if not self.is_population:
            return f'noop, num_params={n}'
        return f'pop_size={self.pop_size}, num_params={n}, mode={"functional" if self.is_functional else "backed"}'
