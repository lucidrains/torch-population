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
    check_pop_size,
    clone_template,
    default,
    encode_buffer_key,
    exists,
    init_module,
    is_int_index,
    is_sequence,
    matches_pattern,
    normalize_index,
    pad_pop,
    reduce_pop,
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

def mean_or_first(t: Tensor, weights: Tensor | None = None) -> Tensor:
    t = t.detach()
    if not t.is_floating_point():
        return t[0]
    return torch.tensordot(weights.to(t), t, dims = 1) if exists(weights) else t.mean(dim = 0)

def pack_param_group(specs: list[TensorSpec], module_params: list[dict[str, Tensor]], *, is_shared: bool) -> Parameter | None:
    if not specs:
        return None
    if len({module_params[0][s.name].dtype for s in specs}) > 1:
        raise ValueError('trainable parameters in a Population must share the same dtype')

    stacked = [torch.stack([mp[s.name].detach() for mp in module_params]) for s in specs]
    if is_shared:
        packed, _ = pack([mean_or_first(t) for t in stacked], '*')
    else:
        packed, _ = pack(stacked, 'p *')
    return Parameter(packed, requires_grad = True)

def _clip_grad_tensors_(
    grads: list[Tensor],
    max_norm: float,
    ord_val: float,
    *,
    indices: list[int] | None = None,
    batched: bool = False,
    error_if_nonfinite: bool = False,
) -> Tensor:
    if batched:
        parts = [
            torch.linalg.vector_norm((g[indices] if exists(indices) else g).flatten(1), ord = ord_val, dim = -1)
            for g in grads
        ]
    else:
        parts = [torch.linalg.vector_norm(g.reshape(-1), ord = ord_val) for g in grads]

    total = parts[0] if len(parts) == 1 else torch.linalg.vector_norm(torch.stack(parts, dim = -1), ord = ord_val, dim = -1)
    if error_if_nonfinite and not torch.isfinite(total).all():
        raise RuntimeError(f'The total norm of order {ord_val} for gradients is non-finite, so it cannot be clipped.')

    coef = (max_norm / (total + 1e-6)).clamp(max = 1.)
    for g in grads:
        scale = coef.reshape(-1, *((1,) * (g.ndim - 1))) if batched else coef
        if batched and exists(indices):
            g[indices] = g[indices] * scale
        else:
            g.mul_(scale)
    return total

def copy_tensor_(dst: Tensor, src: Tensor, kind: str, name: str, *, check_shapes: bool = False):
    if check_shapes and src.shape != dst.shape:
        raise ValueError(f'{kind} {name!r} shape mismatch: {tuple(dst.shape)} vs {tuple(src.shape)}')
    dst.copy_(src.detach().to(dst))

def _expand_row(t: Tensor, pop_size: int, has_pop_dim: bool, *, zero: bool = False) -> Tensor:
    base = t.detach()
    shape = (pop_size, *base.shape[1:]) if has_pop_dim else (pop_size, *base.shape)
    if zero:
        return torch.zeros(shape, device = base.device, dtype = base.dtype)
    return (base.expand(*shape) if has_pop_dim else base.unsqueeze(0).expand(*shape)).clone()

def _expand_opt_entry(opt_store: OptStore, subkey: str, *, is_pop: bool, reset_opt: bool) -> tuple[bool, bool, bool]:
    was_pop = is_pop and opt_store.is_populated(subkey, 1)
    if subkey.startswith('hp/'):
        return was_pop, False, was_pop
    if subkey == 'global/_init':
        return True, reset_opt, True
    if subkey.startswith('slot/'):
        _, slot_name, suffix = subkey.split('/', 2)
        return (suffix == 'pop') and not opt_store.slots[slot_name].shared, reset_opt, was_pop
    if subkey.startswith('aux/'):
        _, aux_key, param_name = subkey.split('/', 2)
        return not opt_store.aux[aux_key][param_name].shared, reset_opt, was_pop
    return was_pop, reset_opt, was_pop

def lerp_weight(weight: float | Tensor, dst: Tensor, indices: list[int] | None, pop_size: int | None):
    if not is_tensor(weight):
        return weight
    w = weight.detach().to(dst)
    if w.ndim == 0:
        return w.item()
    if exists(pop_size) and dst.ndim >= 1 and dst.shape[0] == pop_size:
        w = w[indices] if (exists(indices) and w.shape[0] == pop_size) else w
        return w.reshape(-1, *((1,) * (dst.ndim - 1)))
    return w.mean().item()

def normalize_modules(modules: Module | Iterable[Module | None]) -> tuple[Module, ...]:
    if isinstance(modules, Module):
        return (modules,)
    out = tuple(m for m in modules if exists(m))
    if not all(isinstance(m, Module) for m in out):
        raise TypeError('expected nn.Module or iterable of nn.Module')
    return out

def iter_named_unique(modules: Sequence[Module], kind: Literal['parameters', 'buffers']) -> list[tuple[str, Tensor]]:
    seen: set[int] = set()
    out: list[tuple[str, Tensor]] = []
    multi = len(modules) > 1
    for i, mod in enumerate(modules):
        prefix = f'{i}.' if multi else ''
        for name, tensor in getattr(mod, f'named_{kind}')():
            if id(tensor) in seen:
                continue
            seen.add(id(tensor))
            out.append((f'{prefix}{name}', tensor))
    return out

class _SurrogateLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value: Tensor, grad: Tensor, sizes: tuple[int, ...], shapes: tuple[tuple[int, ...], ...], *targets: Tensor) -> Tensor:
        ctx.save_for_backward(grad)
        ctx.sizes = sizes
        ctx.shapes = shapes
        return value.detach().clone()

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        (grad,) = ctx.saved_tensors
        scale = grad_out.reshape(*grad_out.shape, *((1,) * (grad.ndim - grad_out.ndim)))
        scaled = scale * grad
        if not ctx.sizes:
            return None, None, None, None, scaled
        parts = [t.view(*scaled.shape[:-1], *shape) for t, shape in zip(scaled.split(ctx.sizes, dim = -1), ctx.shapes)]
        return (None, None, None, None, *parts)

# parameter layout schema

class _Schema:
    def __init__(self, ref: Module | Sequence[Module] | None = None, shared = ()):
        self.specs: list[TensorSpec] = []
        self.state_dict_map: list[tuple[str, str, bool]] = []
        self.shared_buffers: frozenset[str] = frozenset()

        mods = normalize_modules(ref) if exists(ref) else ()
        if mods:
            named_params = iter_named_unique(mods, 'parameters')
            named_buffers = iter_named_unique(mods, 'buffers')
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

            self.shared_buffers = frozenset(name for name, _ in named_buffers if matches_pattern(name, shared))
            if len(mods) == 1:
                by_id = {id(p): (name, True) for name, p in named_params}
                by_id.update({id(b): (name, False) for name, b in named_buffers})
                param_names = {s.name for s in self.specs}
                self.state_dict_map = [
                    (sd_key, *by_id.get(id(t), (sd_key, sd_key in param_names)))
                    for sd_key, t in mods[0].state_dict().items()
                ]
            else:
                self.state_dict_map = [(n, n, True) for n, _ in named_params] + [(n, n, False) for n, _ in named_buffers]

        self.by_name = {s.name: s for s in self.specs}
        self.param_names = [s.name for s in self.specs]
        self.trainable = [s for s in self.specs if s.requires_grad]
        self.pop_trainable = [s for s in self.trainable if not s.shared]
        self.shared_trainable = [s for s in self.trainable if s.shared]
        self.trainable_shapes = [s.shape for s in self.trainable]
        self.trainable_sizes = [s.numel for s in self.trainable]
        self.pop_shapes = [s.shape for s in self.pop_trainable]
        self.pop_sizes = [s.numel for s in self.pop_trainable]
        self.shared_shapes = [s.shape for s in self.shared_trainable]
        self.shared_sizes = [s.numel for s in self.shared_trainable]
        self.has_shared = any(s.shared for s in self.specs) or bool(self.shared_buffers)

# unified colocated population parameter container

class PopulationParameters(Module):
    def __init__(self, modules: Module | Iterable[Module | None] | None = None, *, factory: Callable[[], Module] | None = None):
        super().__init__()
        self.register_parameter('_pop_param', None)
        self.register_parameter('_shared_param', None)
        self.store = TensorStore()
        self.opt_stores: dict[str, OptStore] = {}
        self._optimizers: dict[str, PopulationOptimizer] = {}
        self._default_opt_name: str | None = None
        self._active_grads: set[str] | None = None
        self.pop_size: int | None = None
        self._set_raw(_noop_net = None, _noop_nets = (), _template = None, _factory = None)
        self._schema = _Schema()
        self._func_leaves: tuple[Tensor | None, Tensor | None] | None = None
        self._func_params: dict[str, Tensor] | None = None
        if exists(modules):
            self.init_noop(modules, factory = factory)

    def _set_raw(self, **attrs):
        # bypass nn.Module._modules registration for template / noop references
        for key, val in attrs.items():
            object.__setattr__(self, key, val)

    # initialization & cloning

    def init_noop(self, net: Module | Iterable[Module | None], *, factory: Callable[[], Module] | None = None):
        nets = normalize_modules(net)
        self.pop_size = None
        self._set_raw(_noop_net = nets[0] if len(nets) == 1 else None, _noop_nets = nets, _template = None, _factory = factory)
        self._schema = _Schema(nets)
        self._func_leaves = None
        self._func_params = None
        self._active_grads = None

    def init_from_modules(self, modules: list[Module], template: Module, *, shared = (), factory: Callable[[], Module] | None = None):
        ref = modules[0]
        module_params = [dict(m.named_parameters()) for m in modules]
        module_buffers = [dict(m.named_buffers()) for m in modules]
        validate_module_maps('parameter', module_params)
        validate_module_maps('buffer', module_buffers)

        schema = _Schema(ref, shared)
        self.pop_size = len(modules)
        self._set_raw(_noop_net = None, _noop_nets = (), _template = template, **({'_factory': factory} if exists(factory) else {}))
        self._schema = schema
        self._func_leaves = None
        self._func_params = None
        self._active_grads = None

        self._pop_param = pack_param_group(schema.pop_trainable, module_params, is_shared = False)
        self._shared_param = pack_param_group(schema.shared_trainable, module_params, is_shared = True)

        self.store.remove_('frozen/')
        for spec in schema.specs:
            if spec.requires_grad:
                continue
            stacked = torch.stack([mp[spec.name].detach().clone() for mp in module_params])
            self.store.set(f'frozen/{spec.name}', mean_or_first(stacked) if spec.shared else stacked, populated = not spec.shared)

        sd_keys = set(ref.state_dict())
        self.store.remove_('buffer/')
        for name in module_buffers[0]:
            stacked = torch.stack([mb[name].detach().clone() for mb in module_buffers])
            is_shared = name in schema.shared_buffers
            val = mean_or_first(stacked) if is_shared else stacked
            self.store.set(f'buffer/{name}', val, populated = not is_shared, persistent = name in sd_keys, append_repeat = not is_shared)

    def _set_leaves(
        self,
        leaves: tuple[Tensor | None, Tensor | None],
        *,
        functional: bool = False,
        orig: tuple[Tensor | None, Tensor | None] = (None, None),
    ):
        if functional:
            self._func_leaves = leaves
            return
        self._pop_param, self._shared_param = (
            Parameter(cur.detach().clone() if cur is prev else cur.detach(), requires_grad = True) if exists(cur) else None
            for cur, prev in zip(leaves, orig)
        )

    def _spawn(
        self,
        store: TensorStore,
        leaves: tuple[Tensor | None, Tensor | None],
        *,
        pop_size: int | None = None,
        template: Module | None = None,
        functional: bool = False,
        func_params: dict[str, Tensor] | None = None,
        clone_opts: bool = True,
    ) -> PopulationParameters:
        out = PopulationParameters()
        out.pop_size = default(pop_size, self.pop_size)
        out._set_raw(_noop_net = self._noop_net, _noop_nets = self._noop_nets, _template = default(template, self._template), _factory = self._factory)
        out._schema = self._schema
        out.store = store
        if exists(func_params):
            out._func_params = func_params
        else:
            out._set_leaves(leaves, functional = functional)

        if clone_opts:
            out.opt_stores = {k: v.clone_handle(out) for k, v in self.opt_stores.items()}
            out._optimizers = dict(self._optimizers)
            out._default_opt_name = self._default_opt_name
        return out

    def _store_prefixes(self, include_opt: bool, include_buffers: bool = True) -> tuple[str, ...]:
        prefixes = ('frozen/', 'track/', 'track_init/')
        if include_buffers:
            prefixes += ('buffer/',)
        if include_opt:
            prefixes += ('opt/',)
        return prefixes

    def sliced(self, indices: list[int] | None = None, *, clone_opts: bool = False, functional: bool = False) -> PopulationParameters:
        detach = not functional
        store = self.store.clone(detach = detach, prefixes = self._store_prefixes(clone_opts))
        if exists(indices):
            store.keep_(indices, self.pop_size)

        def copy_leaf(leaf: Tensor | None, is_pop: bool) -> Tensor | None:
            if not exists(leaf):
                return None
            if is_pop and exists(indices):
                return slice_pop(leaf, indices)
            return leaf.detach().clone() if detach else leaf.clone()

        leaves = tuple(copy_leaf(leaf, is_pop) for is_pop, leaf in zip((True, False), self._leaf_pair))
        return self._spawn(
            store, leaves,
            pop_size = len(indices) if exists(indices) else self.pop_size,
            template = self._template if functional else deepcopy(self._template),
            functional = functional,
            clone_opts = clone_opts,
        )

    def _clone_noop(self, *, include_opt: bool, copy_func_params: bool = False) -> PopulationParameters:
        out = PopulationParameters()
        out.init_noop(deepcopy(self._noop_nets), factory = self._factory)
        if copy_func_params and exists(self._func_params):
            with torch.no_grad():
                out._copy_member_state_(0, self._func_params, None)
        out.store = self.store.clone(detach = True, prefixes = self._store_prefixes(include_opt))
        if include_opt:
            out.opt_stores = {k: v.clone_handle(out) for k, v in self.opt_stores.items()}
            out._optimizers = dict(self._optimizers)
            out._default_opt_name = self._default_opt_name
        return out

    def clone(self, *, include_opt: bool = True) -> PopulationParameters:
        if self.is_population:
            return self.sliced(clone_opts = include_opt, functional = self.is_functional)
        if not self.is_functional:
            return self._clone_noop(include_opt = include_opt)
        return self._spawn(
            self.store.clone(detach = False, prefixes = self._store_prefixes(include_opt)),
            (None, None),
            func_params = {k: v.clone() for k, v in self._func_params.items()},
            clone_opts = include_opt,
        )

    def detach(self, *, include_opt: bool = True) -> PopulationParameters:
        if self.is_population:
            return self.sliced(clone_opts = include_opt)
        return self._clone_noop(include_opt = include_opt, copy_func_params = True)

    def expand(self, pop_size: int, *, include_opt: bool = True, reset_opt: bool = False) -> PopulationParameters:
        check_pop_size(pop_size)
        if self.is_population and self.pop_size != 1:
            raise ValueError(f'expand() requires a single-model PopulationParameters (noop or pop_size=1), got pop_size={self.pop_size}')
        if not exists(self.template):
            raise ValueError('expand() requires a single-module PopulationParameters')

        if not self.is_population and include_opt:
            for opt in self._optimizers.values():
                opt._sync_noop_device()

        if not self.is_population:
            named_p = self.named_parameters_dict()
            pop_leaf = self.flat_params.unsqueeze(0).expand(pop_size, -1) if self._schema.trainable else None
            shared_leaf = None
        else:
            pop_leaf = self._pop_leaf.expand(pop_size, -1) if exists(self._pop_leaf) else None
            shared_leaf = self._shared_leaf

        store = TensorStore()
        if not self.is_population:
            for spec in self._schema.specs:
                if not spec.requires_grad:
                    store.set(f'frozen/{spec.name}', _expand_row(named_p[spec.name], pop_size, False), populated = True)
            sd_keys = set(self._noop_net.state_dict()) if exists(self._noop_net) else set()
            for name, buf in self.named_buffers_dict().items():
                store.set(f'buffer/{name}', _expand_row(buf, pop_size, False), populated = True, persistent = name in sd_keys, append_repeat = True)
        else:
            for prefix in ('frozen/', 'buffer/'):
                for key, tensor in self.store.items(prefix):
                    full_key = f'{prefix}{key}'
                    is_pop = self.store.is_populated(full_key, 1)
                    val = _expand_row(tensor, pop_size, True) if is_pop else tensor.detach().clone()
                    store.set(
                        full_key, val,
                        populated = is_pop,
                        persistent = encode_buffer_key(full_key) not in self.store._non_persistent_buffers_set,
                        append_repeat = full_key in self.store._repeat_on_append,
                    )

        if include_opt:
            for opt_store in self.opt_stores.values():
                for subkey, tensor in opt_store.items():
                    full_key = opt_store._prefix + subkey
                    should_pop, should_zero, has_dim = _expand_opt_entry(opt_store, subkey, is_pop = self.is_population, reset_opt = reset_opt)
                    val = _expand_row(tensor, pop_size, has_dim, zero = should_zero) if should_pop else (torch.zeros_like(tensor) if should_zero else tensor.detach().clone())
                    store.set(full_key, val, populated = should_pop, append_repeat = full_key in self.store._repeat_on_append)

        return self._spawn(
            store, (pop_leaf, shared_leaf),
            pop_size = pop_size,
            template = self.template,
            functional = True,
            clone_opts = include_opt,
        )

    def _apply(self, fn, recurse = True):
        res = super()._apply(fn, recurse = recurse)
        if exists(self._func_leaves):
            self._func_leaves = tuple(fn(t) if exists(t) else None for t in self._func_leaves)
        if exists(self._func_params):
            self._func_params = {k: fn(v) for k, v in self._func_params.items()}
        return res

    def _load_from_state_dict(self, *args, **kwargs):
        super()._load_from_state_dict(*args, **kwargs)
        for opt in self._optimizers.values():
            opt._noop_cache = None

    # accessors

    @property
    def is_population(self) -> bool:
        return exists(self.pop_size)

    @property
    def is_functional(self) -> bool:
        return exists(self._func_leaves) or exists(self._func_params)

    @property
    def template(self) -> Module:
        return self._template if self.is_population else self._noop_net

    @property
    def trainable_specs(self) -> list[TensorSpec]:
        return self._schema.trainable

    @property
    def _leaf_pair(self) -> tuple[Tensor | None, Tensor | None]:
        return self._func_leaves if exists(self._func_leaves) else (self._pop_param, self._shared_param)

    @property
    def _pop_leaf(self) -> Tensor | None:
        return self._leaf_pair[0]

    @property
    def _shared_leaf(self) -> Tensor | None:
        return self._leaf_pair[1]

    @property
    def _leaves(self) -> list[Tensor]:
        return [t for t in self._leaf_pair if exists(t)]

    @property
    def buffer_names(self) -> list[str]:
        entries = self.store.items('buffer/') if self.is_population else iter_named_unique(self._noop_nets, 'buffers')
        return [name for name, _ in entries]

    @property
    def flat_params(self) -> Tensor:
        if not self.is_population:
            named = self.named_parameters_dict()
            trainable = [named[s.name] for s in self._schema.trainable]
            if len({p.dtype for p in trainable}) > 1:
                raise ValueError('flat_params requires a single trainable parameter dtype')
            return torch.cat([p.reshape(-1) for p in trainable]) if trainable else torch.empty(0)
        return self._pop_leaf if exists(self._pop_leaf) else torch.empty((self.pop_size, 0))

    @property
    def flat_shared_params(self) -> Tensor | None:
        return self._shared_leaf

    def _require_pop(self, op: str, *, backed: bool = False):
        if not self.is_population:
            raise ValueError(f'{op} requires a populated PopulationParameters')
        if backed and self.is_functional:
            raise ValueError(f'{op} requires a detached/backed PopulationParameters; call detach() first')

    def has_param(self, name: str) -> bool:
        return name in self._schema.by_name

    def is_shared_param(self, name: str) -> bool:
        spec = self._schema.by_name.get(name)
        return exists(spec) and spec.shared

    def is_shared_buffer(self, name: str) -> bool:
        return name in self._schema.shared_buffers

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
        schema = self._schema
        specs, shapes, sizes = (
            (schema.pop_trainable, schema.pop_shapes, schema.pop_sizes)
            if self.is_population else
            (schema.trainable, schema.trainable_shapes, schema.trainable_sizes)
        )
        return dict(zip((s.name for s in specs), unpack_tensors(flat, shapes, sizes)))

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
            buffers = {
                name: (buf if self.is_shared_buffer(name) else buf[idx])
                for name, buf in self.store.items('buffer/')
            }

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
                is_shared = self.is_shared_buffer(name)
                if (skip_shared and is_shared) or name not in src_buffers:
                    continue
                target = dst[idx] if (self.is_population and not is_shared) else dst
                copy_tensor_(target, src_buffers[name], 'buffer', name, check_shapes = check_shapes)

    def named_parameters_dict(self, *, track_grads: bool = False) -> dict[str, Tensor]:
        if not self.is_population:
            if exists(self._func_params):
                return dict(self._func_params)
            return dict(iter_named_unique(self._noop_nets, 'parameters'))

        schema = self._schema
        should_hook = track_grads and torch.is_grad_enabled() and not self.is_functional
        if should_hook and not exists(self._active_grads):
            self._active_grads = set()

        # single split per leaf avoids O(L^2) SliceBackward allocations during autograd
        pop_views = iter(unpack_tensors(self._pop_leaf, schema.pop_shapes, schema.pop_sizes)) if exists(self._pop_leaf) else iter(())
        shared_views = iter(unpack_tensors(self._shared_leaf, schema.shared_shapes, schema.shared_sizes)) if exists(self._shared_leaf) else iter(())

        named = {}
        for spec in schema.specs:
            if not spec.requires_grad:
                view = self.store[f'frozen/{spec.name}']
            else:
                view = next(shared_views if spec.shared else pop_views)
                if should_hook and view.requires_grad:
                    view.register_hook(
                        lambda _g, name = spec.name: self._active_grads.add(name) if exists(self._active_grads) else None
                    )
            named[spec.name] = view
        return named

    def named_grads_dict(self) -> dict[str, Tensor | None]:
        if not self.is_population:
            return {name: p.grad for name, p in iter_named_unique(self._noop_nets, 'parameters')}

        grads = {}
        for spec in self._schema.specs:
            leaf = (self._shared_leaf if spec.shared else self._pop_leaf) if spec.requires_grad else None
            grad = leaf.grad if exists(leaf) else None
            inactive = exists(self._active_grads) and spec.name not in self._active_grads
            grads[spec.name] = None if (not exists(grad) or inactive) else slice_spec(spec, grad)
        return grads

    def named_buffers_dict(self) -> dict[str, Tensor]:
        return self.store.as_dict('buffer/') if self.is_population else dict(iter_named_unique(self._noop_nets, 'buffers'))

    def named_optimizer_state(self, opt = None) -> dict[str, dict[str, Tensor]] | None:
        owner = self._opt_owner(opt)
        return owner.named_state(self) if exists(owner) else None

    def _leaf_grads(
        self,
        loss: Tensor,
        *,
        flags: tuple[bool, bool] = (True, True),
        create_graph: bool = False,
        retain_graph: bool | None = None,
    ) -> tuple[Tensor | None, Tensor | None]:
        active = [(i, leaf) for i, (want, leaf) in enumerate(zip(flags, self._leaf_pair)) if want and exists(leaf)]
        if not active:
            return None, None
        raw = torch_grad(
            loss, [leaf for _, leaf in active],
            only_inputs = True,
            create_graph = create_graph,
            retain_graph = default(retain_graph, create_graph),
            allow_unused = True,
        )
        out: list[Tensor | None] = [None, None]
        for (i, _), grad in zip(active, raw):
            out[i] = grad
        return out[0], out[1]

    def _autograd_one(
        self,
        loss: Tensor,
        *,
        active: set[str],
        scoped: bool,
        retain_graph: bool | None,
        create_graph: bool,
        flat: bool,
    ) -> Tensor | dict[str, Tensor]:
        trainable = self._schema.trainable

        if not self.is_population:
            named = self.named_parameters_dict()
            keys = [s.name for s in trainable if s.name in active and named[s.name].requires_grad] if loss.requires_grad else []
            raw = dict(zip(keys, torch_grad(
                loss, [named[k] for k in keys],
                only_inputs = True,
                retain_graph = default(retain_graph, create_graph),
                create_graph = create_graph,
                allow_unused = True,
            ))) if keys else {}
            if flat:
                if not trainable:
                    return torch.empty(0, device = loss.device, dtype = loss.dtype)
                parts = [g if exists(g := raw.get(s.name)) else torch.zeros_like(named[s.name]) for s in trainable]
                if len({t.dtype for t in parts}) > 1:
                    raise ValueError('flat autograd requires a single trainable parameter dtype')
                return torch.cat([t.reshape(-1) for t in parts])
            return {s.name: (g if exists(g := raw.get(s.name)) else torch.zeros_like(named[s.name])) for s in trainable}

        if flat:
            if not exists(self._pop_leaf):
                return torch.empty((self.pop_size, 0), device = loss.device, dtype = loss.dtype)
            pop_specs = self._schema.pop_trainable
            need_pop = loss.requires_grad and any(s.name in active for s in pop_specs)
            pop_grad, _ = self._leaf_grads(loss, flags = (need_pop, False), create_graph = create_graph, retain_graph = retain_graph)
            if not exists(pop_grad):
                return torch.zeros_like(self._pop_leaf)
            if not scoped:
                return pop_grad
            return torch.cat([
                (slice_spec(s, pop_grad) if s.name in active else torch.zeros_like(self._param_view(s))).flatten(1)
                for s in pop_specs
            ], dim = -1)

        flags = (
            loss.requires_grad and any(s.name in active and not s.shared for s in trainable),
            loss.requires_grad and any(s.name in active and s.shared for s in trainable),
        )
        pop_grad, shared_grad = self._leaf_grads(loss, flags = flags, create_graph = create_graph, retain_graph = retain_graph)
        out = {}
        for s in trainable:
            grad = (shared_grad if s.shared else pop_grad) if s.name in active else None
            out[s.name] = slice_spec(s, grad) if exists(grad) else torch.zeros_like(self._param_view(s))
        return out

    def autograd(
        self,
        *losses: Tensor | Sequence[Tensor],
        names: Iterable[str] | None = None,
        retain_graph: bool | None = None,
        create_graph: bool = False,
        flat: bool = False,
    ) -> Tensor | dict[str, Tensor] | tuple[Tensor | dict[str, Tensor], ...]:
        from_seq = len(losses) == 1 and is_sequence(losses[0])
        loss_list = list(losses[0]) if from_seq else list(losses)
        if not loss_list:
            raise ValueError('autograd requires at least one loss tensor')
        active = set(names) if exists(names) else {s.name for s in self._schema.trainable}
        last = len(loss_list) - 1
        results = [
            self._autograd_one(
                loss,
                active = active,
                scoped = exists(names),
                retain_graph = True if i < last else retain_graph,
                create_graph = create_graph,
                flat = flat,
            )
            for i, loss in enumerate(loss_list)
        ]
        return tuple(results) if (from_seq or len(results) > 1) else results[0]

    def _surrogate_noop_targets(self, value: Tensor, flat_grad: Tensor, targets: list[Tensor]) -> Tensor:
        if not targets:
            return value.detach().clone()
        sizes = tuple(p.numel() for p in targets)
        shapes = tuple(tuple(p.shape) for p in targets)
        return _SurrogateLoss.apply(value.detach(), flat_grad, sizes, shapes, *targets)

    def surrogate(
        self,
        value: Tensor,
        grads: Tensor | Mapping[str, Tensor | None],
        *,
        names: Iterable[str] | None = None,
    ) -> Tensor:
        allowed = set(names) if exists(names) else None

        if not self.is_population:
            named = self.named_parameters_dict()
            if is_tensor(grads):
                specs = [s for s in self._schema.trainable if not exists(allowed) or s.name in allowed]
                targets = [named[s.name] for s in specs if named[s.name].requires_grad]
                if len(targets) != len(specs):
                    sizes = [s.numel for s in specs]
                    chunks = grads.split(sizes, dim = -1)
                    kept = [(named[s.name], c) for s, c in zip(specs, chunks) if named[s.name].requires_grad]
                    if not kept:
                        return value.detach().clone()
                    targets = [p for p, _ in kept]
                    grads = torch.cat([c for _, c in kept], dim = -1)
                return self._surrogate_noop_targets(value, grads, targets)

            kept = [
                (named[s.name], g)
                for s in self._schema.trainable
                if (not exists(allowed) or s.name in allowed) and named[s.name].requires_grad and exists(g := grads.get(s.name))
            ]
            if not kept:
                return value.detach().clone()
            return self._surrogate_noop_targets(value, torch.cat([g.reshape(-1) for _, g in kept]), [p for p, _ in kept])

        if is_tensor(grads):
            if not exists(self._pop_leaf):
                return value.detach().clone()
            return _SurrogateLoss.apply(value.detach(), grads, (), (tuple(self._pop_leaf.shape),), self._pop_leaf)

        out = None
        for leaf, specs, is_pop in (
            (self._pop_leaf, self._schema.pop_trainable, True),
            (self._shared_leaf, self._schema.shared_trainable, False),
        ):
            if not exists(leaf) or not any(exists(grads.get(s.name)) and (not exists(allowed) or s.name in allowed) for s in specs):
                continue
            parts = [
                (g if (not exists(allowed) or s.name in allowed) and exists(g := grads.get(s.name)) else torch.zeros_like(self._param_view(s))).flatten(1 if is_pop else 0)
                for s in specs
            ]
            flat_g = torch.cat(parts, dim = -1)
            term = _SurrogateLoss.apply(value.detach() if not exists(out) else torch.zeros_like(value), flat_g, (), (tuple(leaf.shape),), leaf)
            out = term if not exists(out) else (out + term)
        return out if exists(out) else value.detach().clone()

    def reconcile(
        self,
        *losses: Tensor | Sequence[Tensor],
        fn: Callable[..., Tensor | Mapping[str, Tensor]],
        value: Tensor | None = None,
        names: Iterable[str] | None = None,
        flat: bool = True,
        retain_graph: bool | None = None,
        create_graph: bool = False,
    ) -> Tensor:
        from_seq = len(losses) == 1 and is_sequence(losses[0])
        loss_list = list(losses[0]) if from_seq else list(losses)
        if not loss_list:
            raise ValueError('reconcile requires at least one loss tensor')

        out_val = default(value, loss_list[0])
        if not torch.is_grad_enabled() or not any(loss.requires_grad for loss in loss_list):
            return out_val

        if not self.is_population and flat:
            named = self.named_parameters_dict()
            allowed = set(names) if exists(names) else None
            cand = [s for s in self._schema.trainable if (not exists(allowed) or s.name in allowed) and named[s.name].requires_grad]
            last = len(loss_list) - 1
            raws = [
                dict(zip((s.name for s in cand), torch_grad(
                    loss, [named[s.name] for s in cand],
                    only_inputs = True,
                    retain_graph = True if i < last else default(retain_graph, create_graph),
                    create_graph = create_graph,
                    allow_unused = True,
                ))) if (cand and loss.requires_grad) else {}
                for i, loss in enumerate(loss_list)
            ]
            used = [s for s in cand if any(exists(r.get(s.name)) for r in raws)]
            if not used:
                return out_val
            flat_grads = tuple(
                torch.cat([(g if exists(g := r.get(s.name)) else torch.zeros_like(named[s.name])).reshape(-1) for s in used])
                for r in raws
            )
            return self._surrogate_noop_targets(out_val, fn(*flat_grads), [named[s.name] for s in used])

        grads = self.autograd(*loss_list, names = names, retain_graph = retain_graph, create_graph = create_graph, flat = flat)
        combined = fn(*(grads if (from_seq or len(loss_list) > 1) else (grads,)))
        return self.surrogate(out_val, combined, names = names)

    @torch.no_grad()
    def set_grads_(
        self,
        grads: Tensor | Mapping[str, Tensor | None],
        *,
        accumulate: bool = False,
        names: Iterable[str] | None = None,
    ) -> PopulationParameters:
        allowed = set(names) if exists(names) else None

        def assign(param: Tensor, g: Tensor):
            g_det = g.detach().to(param)
            if accumulate and exists(param.grad):
                param.grad.add_(g_det)
            else:
                param.grad = g_det.clone()

        if not self.is_population:
            named = self.named_parameters_dict()
            grad_map = self.unpack_flat(grads) if is_tensor(grads) else grads
            for s in self._schema.trainable:
                if (exists(allowed) and s.name not in allowed) or not exists(g := grad_map.get(s.name)):
                    continue
                assign(named[s.name], g)
            return self

        if is_tensor(grads):
            if exists(self._pop_leaf):
                assign(self._pop_leaf, grads)
            return self

        for leaf, specs, is_pop in (
            (self._pop_leaf, self._schema.pop_trainable, True),
            (self._shared_leaf, self._schema.shared_trainable, False),
        ):
            if not exists(leaf) or not any(exists(grads.get(s.name)) and (not exists(allowed) or s.name in allowed) for s in specs):
                continue
            if not exists(leaf.grad) or not accumulate:
                leaf.grad = torch.zeros_like(leaf)
            for s in specs:
                if (exists(allowed) and s.name not in allowed) or not exists(g := grads.get(s.name)):
                    continue
                dst = slice_spec(s, leaf.grad)
                if accumulate:
                    dst.add_(g.detach().to(dst))
                else:
                    dst.copy_(g.detach().to(dst))
        return self

    def _default_device(self) -> torch.device:
        for t in self._leaf_pair:
            if exists(t):
                return t.device
        for kind in ('parameters', 'buffers'):
            for _, t in iter_named_unique(self._noop_nets, kind):
                return t.device
        return torch.device('cpu')

    @torch.no_grad()
    def clip_grad_norm_(
        self,
        max_norm: float,
        norm_type: float = 2.0,
        *,
        members = None,
        names = None,
        error_if_nonfinite: bool = False,
    ) -> Tensor:
        max_norm, ord_val = float(max_norm), float(norm_type)

        if not self.is_population:
            if exists(members) and not self.resolve_members(members):
                return torch.tensor(0., device = self._default_device())
            named = self.named_parameters_dict()
            grads = [
                named[s.name].grad for s in self._schema.trainable
                if (not exists(names) or matches_pattern(s.name, names)) and exists(named[s.name].grad)
            ]
            if not grads:
                return torch.tensor(0., device = self._default_device())
            return _clip_grad_tensors_(grads, max_norm, ord_val, error_if_nonfinite = error_if_nonfinite)

        indices = list(dict.fromkeys(self.resolve_members(members))) if exists(members) else None
        if exists(indices) and not indices:
            return torch.empty(0, device = self._default_device())
        num_active = len(indices) if exists(indices) else self.pop_size
        scoped = exists(names) or exists(self._active_grads)
        named_grads = self.named_grads_dict() if scoped else None

        def select_grads(leaf: Tensor | None, specs: list[TensorSpec]) -> list[Tensor]:
            if not exists(leaf) or not exists(leaf.grad):
                return []
            if not scoped:
                return [leaf.grad]
            return [g for s in specs if (not exists(names) or matches_pattern(s.name, names)) and exists(g := named_grads.get(s.name))]

        # clip per-member gradients independently; shared gradients clip on their own tensor
        pop_grads = select_grads(self._pop_leaf, self._schema.pop_trainable)
        shared_grads = select_grads(self._shared_leaf, self._schema.shared_trainable)

        pop_norm = _clip_grad_tensors_(pop_grads, max_norm, ord_val, indices = indices, batched = True, error_if_nonfinite = error_if_nonfinite) if pop_grads else None
        shared_norm = _clip_grad_tensors_(shared_grads, max_norm, ord_val, error_if_nonfinite = error_if_nonfinite) if shared_grads else None

        if exists(self._pop_leaf):
            return default(pop_norm, torch.zeros(num_active, device = self._pop_leaf.device, dtype = self._pop_leaf.dtype))
        return default(shared_norm, torch.tensor(0., device = self._default_device()))

    def zero_grad_(self, members = None, set_to_none: bool = True):
        indices = self.resolve_members(members) if exists(members) else None
        if exists(indices) and not indices:
            return

        if exists(indices) and self.is_population:
            with torch.no_grad():
                if exists(self._pop_leaf) and exists(self._pop_leaf.grad):
                    self._pop_leaf.grad[indices] = 0.
            return

        params = self._leaves if self.is_population else [p for _, p in iter_named_unique(self._noop_nets, 'parameters')]
        for param in params:
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
                sd_key: table[canonical]
                for sd_key, canonical, is_param in self._schema.state_dict_map
                if canonical in (table := named_params if is_param else named_buffers)
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

    def reduce_dict(self, reduction = 'mean') -> dict:
        if not self.is_population:
            return self.individual(0, clone = True)

        def reduce_tensor(t: Tensor, is_shared: bool) -> Tensor:
            t = t.detach()
            return t.clone() if is_shared else reduce_pop(t, reduction).clone()

        named_params = {s.name: reduce_tensor(self._param_view(s), s.shared) for s in self._schema.specs}
        named_buffers = {name: reduce_tensor(buf, self.is_shared_buffer(name)) for name, buf in self.store.items('buffer/')}
        return self._bundle(named_params, named_buffers, include_state_dict = True)

    def reduce(self, reduction = 'mean', *, training: bool | None = None) -> Module:
        if not self.is_population:
            if not exists(self._noop_net):
                raise ValueError('reduce() requires a single-module PopulationParameters')
            cloned = deepcopy(self._noop_net)
            if exists(training):
                cloned.train(training)
            return cloned

        reduced = self.reduce_dict(reduction = reduction)
        return clone_template(self._template, reduced['named_parameters'], reduced['named_buffers'], training = training)

    def mean_dict(self, weights: Tensor | Sequence[float] | None = None) -> dict:
        if not self.is_population:
            return self.individual(0, clone = True)

        weight_vec = None
        if exists(weights):
            weight_vec = weights.detach().float() if is_tensor(weights) else torch.tensor([float(x) for x in weights])
            if weight_vec.shape != (self.pop_size,):
                raise ValueError(f'weights must have shape ({self.pop_size},), got {tuple(weight_vec.shape)}')
        return self.reduce_dict(lambda t: mean_or_first(t, weight_vec))

    def mean(self, weights: Tensor | Sequence[float] | None = None, *, training: bool | None = None) -> Module:
        if not self.is_population:
            return self.reduce(training = training)
        avg = self.mean_dict(weights = weights)
        return clone_template(self._template, avg['named_parameters'], avg['named_buffers'], training = training)

    def tracked(self, name: str) -> Tensor | None:
        return self.store.get(f'track/{name}')

    @torch.no_grad()
    def clear_tracked(self, name: str | None = None, *, members = None) -> PopulationParameters:
        indices = self.resolve_members(members) if (self.is_population and exists(members)) else None
        keys = (name,) if exists(name) else None
        for prefix in ('track/', 'track_init/'):
            self.store.clear_(prefix, keys = keys, indices = indices, pop_size = self.pop_size, zero = exists(indices))
        return self

    @torch.no_grad()
    def _update_track_(
        self,
        name: str,
        value: Tensor | float | bool | Sequence,
        *,
        decay: float | Tensor | None = None,
        members = None,
    ) -> Tensor:
        src = value.detach() if is_tensor(value) else torch.as_tensor(value, device = self._default_device())
        key, init_key = f'track/{name}', f'track_init/{name}'
        indices = self.resolve_members(members) if (self.is_population and exists(members)) else None
        if exists(indices) and not indices:
            return self.store.get(key)

        cur = self.store.get(key)
        is_pop = self.is_population and (exists(indices) or (src.ndim >= 1 and src.shape[0] == self.pop_size))

        if not exists(cur):
            if is_pop and exists(indices):
                has_full = src.ndim >= 1 and src.shape[0] == self.pop_size
                has_sub = not has_full and src.ndim >= 1 and src.shape[0] == len(indices)
                row_shape = src.shape[1:] if (has_full or has_sub) else src.shape
                buf = torch.zeros((self.pop_size, *row_shape), device = src.device, dtype = src.dtype)
                buf[indices] = src[indices] if has_full else src
                init = torch.zeros(self.pop_size, device = src.device, dtype = torch.bool)
                init[indices] = True
            else:
                buf = src.clone()
                init = torch.ones(self.pop_size if is_pop else (), device = src.device, dtype = torch.bool)
            self.store.set(key, buf, populated = is_pop)
            self.store.set(init_key, init, populated = is_pop)
            return buf

        if not exists(decay) and not (is_pop and exists(indices)):
            if cur.shape == src.shape and cur.dtype == src.dtype and cur.device == src.device:
                cur.copy_(src)
                self.store[init_key].fill_(True)
                return cur
            buf = src.clone()
            init = torch.ones(self.pop_size if is_pop else (), device = src.device, dtype = torch.bool)
            self.store.set(key, buf, populated = is_pop)
            self.store.set(init_key, init, populated = is_pop)
            return buf

        # sync tracked buffer device (and float dtype on EMA updates, e.g. after noop net.to())
        target_dtype = src.dtype if (exists(decay) and cur.is_floating_point() and src.is_floating_point()) else cur.dtype
        if cur.device != src.device or cur.dtype != target_dtype:
            cur = self.store.set(key, cur.to(device = src.device, dtype = target_dtype), populated = is_pop)
        init = self.store[init_key]
        if init.device != src.device:
            init = self.store.set(init_key, init.to(device = src.device), populated = is_pop)

        if not exists(decay):
            cur[indices] = (src[indices] if src.shape == cur.shape else src).to(cur)
            init[indices] = True
            return cur

        alpha_raw = 1. - decay
        if is_pop and exists(indices):
            s = (src[indices] if src.shape == cur.shape else src).to(cur)
            fresh = ~init[indices]
            updated = cur[indices].lerp(s, lerp_weight(alpha_raw, cur, indices, self.pop_size))
            if fresh.any():
                updated = torch.where(fresh.reshape(-1, *((1,) * (s.ndim - 1))), s, updated)
            cur[indices] = updated
            init[indices] = True
        elif is_pop:
            s = src.to(cur)
            fresh = ~init
            cur.lerp_(s, lerp_weight(alpha_raw, cur, None, self.pop_size))
            if fresh.any():
                cur[fresh] = s[fresh]
            init.fill_(True)
        else:
            s = src.to(cur)
            if not bool(init):
                cur.copy_(s)
                init.fill_(True)
            else:
                cur.lerp_(s, lerp_weight(alpha_raw, cur, None, None))
        return cur

    def track_(self, name: str, value: Tensor | float | bool | Sequence, *, members = None) -> Tensor:
        return self._update_track_(name, value, decay = None, members = members)

    def ema_(self, name: str, value: Tensor | float | Sequence, decay: float | Tensor = 0.99, *, members = None) -> Tensor:
        return self._update_track_(name, value, decay = decay, members = members)

    @torch.no_grad()
    def lerp_(self, other, weight: float | Tensor, *, members = None) -> PopulationParameters:
        other_pp = other if isinstance(other, PopulationParameters) else getattr(other, 'population_parameters', None)
        is_multi_view = hasattr(other, '_pop') and hasattr(other, '_indices') and not other.is_single
        indices = self.resolve_members(members) if (self.is_population and exists(members)) else None
        if exists(indices) and not indices:
            return self

        # resolve populated vectorized source (matching Population/PopulationParameters, sized subset, or multi-index PopulationView)
        src_pp = src_indices = None
        if self.is_population:
            dst_len = len(indices) if exists(indices) else self.pop_size
            if is_multi_view and other._pop.is_population:
                src_pp, src_indices = other._pop.population_parameters, other._indices
                if len(src_indices) != dst_len:
                    raise ValueError(f'lerp_ source size ({len(src_indices)}) does not match target ({dst_len})')
            elif isinstance(other_pp, PopulationParameters) and other_pp.is_population:
                if other_pp.pop_size == self.pop_size:
                    src_pp, src_indices = other_pp, indices
                elif exists(indices) and other_pp.pop_size == dst_len:
                    src_pp, src_indices = other_pp, list(range(other_pp.pop_size))

        # fast vectorized path (aliasing-safe when src_indices is a list of indices)
        if exists(src_pp) and [s.name for s in self._schema.pop_trainable] == [s.name for s in src_pp._schema.pop_trainable]:
            pairs = [
                (dst, src, is_pop)
                for is_pop, (dst, src) in zip((True, False), zip(self._leaf_pair, src_pp._leaf_pair))
                if exists(dst) and exists(src)
            ]
            for prefix in ('buffer/', 'frozen/'):
                for key, dst in self.store.items(prefix):
                    if exists(src := src_pp.store.get(f'{prefix}{key}')):
                        pairs.append((dst, src, self.store.is_populated(f'{prefix}{key}', self.pop_size)))

            for dst, src, is_pop in pairs:
                if not dst.is_floating_point():
                    continue
                if not is_pop:
                    if exists(indices) or is_multi_view:
                        continue
                    dst.lerp_(src.detach().to(dst), lerp_weight(weight, dst, None, None))
                elif exists(indices):
                    src_t = (src[src_indices] if exists(src_indices) else src).detach().to(dst)
                    dst[indices] = dst[indices].lerp(src_t, lerp_weight(weight, dst, indices, self.pop_size))
                else:
                    src_t = (src[src_indices] if exists(src_indices) else src).detach().to(dst)
                    dst.lerp_(src_t, lerp_weight(weight, dst, None, self.pop_size))
            return self

        # noop-to-noop matching path
        if not self.is_population and isinstance(other_pp, PopulationParameters) and not other_pp.is_population:
            other_p, other_b = other_pp.named_parameters_dict(), other_pp.named_buffers_dict()
            for dst, src in (
                *((dst, other_p[k]) for k, dst in self.named_parameters_dict().items() if k in other_p),
                *((dst, other_b[k]) for k, dst in self.named_buffers_dict().items() if k in other_b),
            ):
                if dst.is_floating_point():
                    dst.lerp_(src.detach().to(dst), lerp_weight(weight, dst, None, None))
            return self

        def weight_at(pos: int, dst_i: int) -> float:
            if not is_tensor(weight):
                return weight
            if weight.ndim == 0:
                return weight.item()
            return weight[dst_i if (self.is_population and weight.shape[0] == self.pop_size) else pos].item()

        # general path: mismatched-layout multi-view, or single-member source (Module, single view, individual dict, 1-member Population)
        if is_multi_view:
            v_pp, v_indices = other._pop.population_parameters, other._indices
            dst_indices = indices if exists(indices) else list(range(len(self)))
            if len(v_indices) != len(dst_indices):
                raise ValueError(f'lerp_ source size ({len(v_indices)}) does not match target ({len(dst_indices)})')
            sources = [v_pp._member_tensors(src_i, clone = (v_pp is self)) for src_i in v_indices]
            for pos, (dst_i, (sp, sb)) in enumerate(zip(dst_indices, sources)):
                self._lerp_member_(dst_i, sp, sb, weight = weight_at(pos, dst_i))
            return self

        src_p, src_b = self._extract_source(other)
        dst_indices = indices if exists(indices) else (list(range(self.pop_size)) if self.is_population else [0])
        for pos, dst_i in enumerate(dst_indices):
            self._lerp_member_(dst_i, src_p, src_b, weight = weight_at(pos, dst_i))
        return self

    def _lerp_member_(self, idx: int, src_p: Mapping[str, Tensor], src_b: Mapping[str, Tensor] | None, *, weight: float):
        for name, dst in self.named_parameters_dict().items():
            spec = self._schema.by_name[name]
            if (self.is_population and spec.shared) or name not in src_p or not dst.is_floating_point():
                continue
            target = dst[idx] if self.is_population else dst
            target.lerp_(src_p[name].detach().to(target), weight)
        if exists(src_b):
            for name, dst in self.named_buffers_dict().items():
                if (self.is_population and self.is_shared_buffer(name)) or name not in src_b or not dst.is_floating_point():
                    continue
                target = dst[idx] if self.is_population else dst
                target.lerp_(src_b[name].detach().to(target), weight)

    @torch.no_grad()
    def copy_from_(
        self,
        other: PopulationParameters,
        *,
        members = None,
        include_opt: bool = True,
        include_buffers: bool = True,
    ) -> PopulationParameters:
        other = getattr(other, 'population_parameters', other)
        if not self.is_population:
            self._copy_member_state_(0, other.named_parameters_dict(), other.named_buffers_dict() if include_buffers else None)
            prefixes = ('track/', 'track_init/') + (('opt/',) if include_opt else ())
            self.store.copy_from_(other.store, prefixes = prefixes)
            if include_opt:
                for opt in self._optimizers.values():
                    opt._noop_cache = None
            return self

        indices = self.resolve_members(members) if exists(members) else None
        if exists(indices) and not indices:
            return self

        for is_pop, (dst, src) in zip((True, False), zip(self._leaf_pair, other._leaf_pair)):
            if not (exists(dst) and exists(src)):
                continue
            if not exists(indices):
                dst.copy_(src.detach().to(dst))
            elif is_pop:
                dst[indices] = src[indices].detach().to(dst)

        self.store.copy_from_(other.store, prefixes = self._store_prefixes(include_opt, include_buffers), indices = indices, pop_size = self.pop_size)
        return self

    def _on_populate_tracks_(self, pop_size: int, *, init, preserve_first: bool):
        for name, val in list(self.store.items('track/')):
            init_flag = self.store[f'track_init/{name}']
            if init == 'clone':
                buf = val.unsqueeze(0).expand(pop_size, *val.shape).clone()
                flags = init_flag.unsqueeze(0).expand(pop_size).clone()
            else:
                buf = torch.zeros((pop_size, *val.shape), device = val.device, dtype = val.dtype)
                flags = torch.zeros(pop_size, device = init_flag.device, dtype = torch.bool)
                if preserve_first:
                    buf[0].copy_(val)
                    flags[0].copy_(init_flag)
            self.store.set(f'track/{name}', buf, populated = True)
            self.store.set(f'track_init/{name}', flags, populated = True)

    def _reset_opts_(self, indices: list[int] | None):
        for opt in self._optimizers.values():
            opt._noop_cache = None
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
            self.copy_from_(ref_pp, members = indices, include_opt = False, include_buffers = reset_buffers)
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
                    if not exists(self._noop_net):
                        raise ValueError('reset_() requires a single-module PopulationParameters')
                    fresh = deepcopy(self._noop_net)
                    init_module(fresh, init)
                else:
                    fresh = clone_template(self._template, *self._member_tensors(i), training = True)
                    init_module(fresh, init)
                src_b = dict(fresh.named_buffers()) if reset_buffers else None
                self._copy_member_state_(i, dict(fresh.named_parameters()), src_b, skip_shared = self.is_population)

        reset_idx = indices if (self.is_population and exists(selector)) else None
        if reset_opt:
            self._reset_opts_(reset_idx)
        for prefix in ('track/', 'track_init/'):
            self.store.zero_(prefix, indices = reset_idx, pop_size = self.pop_size)
        return self

    def _resolve_single_source(self, source) -> tuple[PopulationParameters, int] | None:
        if is_int_index(source):
            return self, self._normalize_index(source)
        src_pp = source if isinstance(source, PopulationParameters) else getattr(source, 'population_parameters', None)
        if isinstance(src_pp, PopulationParameters):
            if src_pp.is_population and src_pp.pop_size != 1:
                raise ValueError(f'cannot extract a single member from population of size {src_pp.pop_size}')
            return src_pp, 0
        if hasattr(source, '_pop') and hasattr(source, 'is_single'):
            if not source.is_single:
                raise ValueError(f'cannot extract a single member from view of size {len(source)}')
            return source._pop.population_parameters, source._index
        return None

    def _extract_source(self, source):
        if exists(resolved := self._resolve_single_source(source)):
            return resolved[0]._member_tensors(resolved[1], clone = True)
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
            for prefix in ('track/', 'track_init/'):
                self.store.zero_(prefix, indices = [idx], pop_size = self.pop_size)
            return self

        if exists(resolved := self._resolve_single_source(source)):
            src_pp, src_idx = resolved
            for prefix in ('track/', 'track_init/'):
                for k, dst_buf in self.store.items(prefix):
                    src_buf = src_pp.store.get(f'{prefix}{k}')
                    if not exists(src_buf) or not self.store.is_populated(f'{prefix}{k}', self.pop_size):
                        continue
                    src_row = src_buf[src_idx] if src_pp.store.is_populated(f'{prefix}{k}', src_pp.pop_size) else src_buf
                    dst_buf[idx] = src_row.detach().to(dst_buf)
            src_opts = src_pp._optimizers
            for name, opt in self._optimizers.items():
                if name in src_opts:
                    opt._copy_from_other_opt_([idx], src_opts[name], [src_idx])
        return self

    def _finish_resize(self, new_size: int):
        self.pop_size = new_size
        self._active_grads = None
        for opt in self._optimizers.values():
            if opt._pp is self:
                opt._sync_optimizer_state()

    @torch.no_grad()
    def keep_(self, members) -> PopulationParameters:
        self._require_pop('keep_()', backed = True)
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
        self._require_pop('append_()', backed = True)
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
        mode = 'functional' if self.is_functional else 'backed'
        if not self.is_population:
            return f'noop, num_params={num_params}' + (f', mode={mode}' if self.is_functional else '')
        return f'pop_size={self.pop_size}, num_params={num_params}, mode={mode}'
