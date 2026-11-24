from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable, Mapping
from contextlib import nullcontext
from typing import TYPE_CHECKING, Literal

import torch
from einops import pack, repeat
from torch import Tensor, is_tensor
from torch.autograd import grad as torch_grad
from torch.nn import Module, Parameter
from torch.optim import Optimizer
from torch.overrides import TorchFunctionMode

from torch_population.parameters import PopulationParameters
from torch_population.store import (
    OptStore,
    ParamSlot,
    TensorSpec,
    default,
    exists,
    is_sequence,
    reduce_pop,
    slice_spec,
)

if TYPE_CHECKING:
    from torch_population.torch_population import Population

# differentiable inner-step guard:
# 1. optionally detach second-moment preconditioner (grad * grad) for clean meta-gradients
# 2. safe sqrt backward to avoid 0 * inf = NaN at zero-grad entries

class _SafeSqrt(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        out = x.sqrt()
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, grad_out: Tensor) -> Tensor:
        (out,) = ctx.saved_tensors
        positive = out > 0
        denom = torch.where(positive, 2. * out, torch.ones_like(out))
        return torch.where(positive, grad_out / denom, torch.zeros_like(grad_out))

class _DifferentiableStepMode(TorchFunctionMode):
    def __init__(self, detach_preconditioner: bool = True):
        self.detach_preconditioner = detach_preconditioner

    def __torch_function__(self, func, types, args = (), kwargs = None):
        kwargs = kwargs or {}
        if self.detach_preconditioner:
            if func in (Tensor.addcmul_, torch.addcmul, Tensor.addcmul) and len(args) >= 3 and args[1] is args[2]:
                return func(args[0], args[1].detach(), args[2].detach(), *args[3:], **kwargs)
            if func in (torch.square, Tensor.square):
                return func(args[0].detach(), **kwargs)
            if func in (torch.mul, Tensor.mul) and len(args) == 2 and args[0] is args[1]:
                return func(args[0].detach(), args[1].detach(), **kwargs)
            if func in (torch.pow, Tensor.pow) and len(args) == 2 and is_tensor(args[0]) and args[1] == 2:
                return func(args[0].detach(), args[1], **kwargs)
        if not kwargs and args and is_tensor(args[0]) and args[0].requires_grad:
            if func in (torch.sqrt, Tensor.sqrt):
                return _SafeSqrt.apply(args[0])
            if func is Tensor.sqrt_:
                return args[0].copy_(_SafeSqrt.apply(args[0].clone()))
        return func(*args, **kwargs)

# helpers

def parse_shared_keys(shared) -> tuple[bool, set[str]]:
    if isinstance(shared, bool):
        return shared, set()
    if isinstance(shared, str):
        return False, {shared}
    items = list(shared)
    if not all(isinstance(k, str) for k in items):
        raise TypeError('shared state keys must be strings')
    return False, set(items)

def inspect_optimizer_signature(opt_cls) -> tuple[Mapping, set[str] | None]:
    try:
        target = opt_cls.__init__ if isinstance(opt_cls, type) else opt_cls
        params = inspect.signature(target).parameters
    except (TypeError, ValueError):
        return {}, None
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return params, None
    accepted = {k for k, p in params.items() if p.kind.name in ('POSITIONAL_OR_KEYWORD', 'KEYWORD_ONLY')}
    return params, accepted

def filter_optimizer_kwargs(kwargs: dict, accepted: set[str] | None, *, differentiable: bool = False) -> dict:
    out = {k: v for k, v in kwargs.items() if not exists(accepted) or k in accepted}
    if not differentiable:
        return out
    if not exists(accepted) or 'differentiable' in accepted:
        out['differentiable'] = True
    for flag in ('foreach', 'fused'):
        if exists(accepted) and flag in accepted:
            out[flag] = False
    return out

def work_tensor(t: Tensor, inplace: bool, create_graph: bool) -> Tensor:
    return t.detach() if inplace else (t.clone() if create_graph else t.detach().clone())

def match_grad(spec: TensorSpec, grad: Tensor, active_indices: list[int] | None, pop_size: int | None) -> Tensor:
    if not exists(pop_size):
        return grad
    if spec.shared:
        if grad.ndim == len(spec.shape):
            return grad
        return (grad[active_indices] if exists(active_indices) else grad).sum(dim = 0)
    if not exists(active_indices):
        return grad
    if grad.ndim == len(spec.shape) + 1 and grad.shape[0] in (pop_size, len(active_indices)):
        return grad[active_indices] if grad.shape[0] == pop_size else grad
    if len(active_indices) == 1 and tuple(grad.shape) == spec.shape:
        return grad.unsqueeze(0)
    raise ValueError(f'grad shape {tuple(grad.shape)} incompatible with {spec.name!r} {spec.shape}')

# hyperparameters (scalar or per-individual 1D tensor)

def is_numeric_sequence(v) -> bool:
    return is_sequence(v) and bool(v) and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v)

def register_hyperparams(store: OptStore, pop_size: int | None, opt_kwargs: dict, sig_params: Mapping, ref: Tensor):
    coerced, probe = {}, {}
    for key, val in opt_kwargs.items():
        sig_param = sig_params.get(key)
        if exists(sig_param) and isinstance(sig_param.default, tuple) and is_sequence(val):
            coerced[key] = probe[key] = tuple(val)
            continue
        if not is_tensor(val) and not is_numeric_sequence(val):
            coerced[key] = probe[key] = val
            continue

        if not is_tensor(val):
            val = torch.tensor([float(x) for x in val], device = ref.device, dtype = ref.dtype)
        tensor = val.detach().to(device = ref.device, dtype = ref.dtype if val.is_floating_point() else val.dtype)

        if tensor.ndim == 0 or (not exists(pop_size) and tensor.shape == (1,)):
            coerced[key] = probe[key] = tensor.item()
            continue
        if not exists(pop_size) or tensor.shape != (pop_size,):
            raise ValueError(f'per-individual {key} requires shape ({pop_size},), got {tuple(tensor.shape)}')

        coerced[key] = store.set(f'hp/{key}', tensor.clone(), populated = True, append_repeat = True)
        probe[key] = coerced[key][0].item()
    return coerced, probe

def copy_hyperparams_(dst_store: OptStore, src_kwargs: dict, dst_indices: list[int], src_indices: list[int] | None = None):
    for key, dst in dst_store.items('hp/'):
        if key not in src_kwargs or not exists(val := src_kwargs[key]):
            continue
        if is_tensor(val) and exists(src_indices):
            dst[dst_indices] = val[src_indices].to(dst)
        else:
            dst[dst_indices] = val.to(dst) if is_tensor(val) else float(val)

# optimizer probing & batched execution eligibility

def _step_probe_pair(opt_cls, opt_args: tuple, opt_kwargs: dict, ref: Tensor, *, hetero: bool = False, accepted: set[str] | None = None) -> bool:
    if hetero and exists(accepted) and 'differentiable' not in accepted:
        return False

    try:
        r0, r1 = torch.randn(2, 3, 2, device = ref.device, dtype = ref.dtype)
        o0, o1 = opt_cls([r0], *opt_args, **opt_kwargs), opt_cls([r1], *opt_args, **opt_kwargs)

        if hetero:
            # warm up member 1 while keeping member 0 freshly reset
            for _ in range(3):
                r1.grad = torch.randn_like(r1)
                o1.step()

        flat = torch.stack((r0.flatten(), r1.flatten()))
        batch_kw = filter_optimizer_kwargs(opt_kwargs, accepted, differentiable = True) if hetero else opt_kwargs
        batch_opt = opt_cls([flat], *opt_args, **batch_kw)

        if hetero:
            state: dict[str, Tensor] = {}
            for key, val in o1.state.get(r1, {}).items():
                if not is_tensor(val):
                    return False
                if val.shape == r1.shape:
                    state[key] = torch.stack((torch.zeros_like(val).flatten(), val.flatten()))
                elif val.ndim == 0:
                    state[key] = torch.stack((torch.zeros_like(val), val)).unsqueeze(-1).to(ref.device)
                else:
                    return False
            batch_opt.state[flat] = state

        for _ in range(2):
            g0, g1 = torch.randn(2, 3, 2, device = ref.device, dtype = ref.dtype)
            for p, g, o in ((r0, g0, o0), (r1, g1, o1), (flat, torch.stack((g0.flatten(), g1.flatten())), batch_opt)):
                p.grad = g
                o.step()

        if not hetero:
            if not all(not is_tensor(v) or v.shape in (flat.shape, ()) for v in batch_opt.state.get(flat, {}).values()):
                return False

        return torch.allclose(flat.view(2, 3, 2), torch.stack((r0, r1)), atol = 1e-5, rtol = 1e-4)
    except Exception:
        return False

@torch.no_grad()
def probe_optimizer(opt_cls, opt_args: tuple, opt_kwargs: dict, specs: list[TensorSpec], named: dict[str, Tensor], accepted: set[str] | None, *, test_batch: bool):
    with torch.random.fork_rng():
        torch.manual_seed(0)
        dummies = [torch.zeros(s.shape, device = named[s.name].device, dtype = named[s.name].dtype) for s in specs]
        for dummy in dummies:
            dummy.grad = torch.ones_like(dummy)

        probe = opt_cls(dummies, *opt_args, **opt_kwargs)
        probe.step()
        probe.step()

        defaults = {k: v for k, v in probe.param_groups[0].items() if k != 'params'}
        state = {s.name: {k: v for k, v in probe.state.get(d, {}).items() if is_tensor(v)} for s, d in zip(specs, dummies)}

        ref = named[specs[0].name]
        can_batch = _step_probe_pair(opt_cls, opt_args, opt_kwargs, ref) if test_batch else False
        can_batch_hetero = _step_probe_pair(opt_cls, opt_args, opt_kwargs, ref, hetero = True, accepted = accepted) if can_batch else False

    return defaults, state, can_batch, can_batch_hetero

def analyze_batched_state(store: OptStore, specs: list[TensorSpec], *, allow_hetero: bool) -> tuple[Literal['uniform', 'hetero'], bool] | None:
    names = [s.name for s in specs]
    if any(slot.pop_names != names for slot in store.slots.values()):
        return None

    flags = store['global/_init'][:, [s.trainable_index for s in specs]]
    row_init = flags[:, 0]
    if not (flags == row_init[:, None]).all():
        return None
    if not flags.any():
        return 'uniform', False

    first = names[0]
    aux_keys = store.aux_keys_for(first)
    if any(store.aux_keys_for(s.name) != aux_keys for s in specs[1:]):
        return None

    all_aux_uniform = True
    for aux_key in aux_keys:
        meta = store.aux[aux_key][first]
        ref = store[f'aux/{aux_key}/{first}']
        if ref.ndim != (0 if meta.shared else 1):
            return None
        if any((t := store[f'aux/{aux_key}/{s.name}']).shape != ref.shape or not (t == ref).all() for s in specs[1:]):
            return None
        if not meta.shared and not (ref == ref[0]).all():
            all_aux_uniform = False

    if row_init.all() and all_aux_uniform:
        return 'uniform', True
    if not allow_hetero:
        return None
    if not row_init.all() and (any(slot.shared for slot in store.slots.values()) or any(store.aux[k][first].shared for k in aux_keys)):
        return None
    return 'hetero', True

# colocated population optimizer

class PopulationOptimizer(Optimizer):
    def __init__(
        self,
        population: Population,
        opt_cls: type[Optimizer] | Callable[..., Optimizer],
        *opt_args,
        shared: bool | str | Iterable[str] = False,
        shared_reduce = 'mean',
        per_individual: bool = False,
        name: str | None = None,
        **opt_kwargs,
    ):
        from torch_population.torch_population import Population

        if not isinstance(population, Population):
            raise TypeError(f'PopulationOptimizer requires a Population instance, got {type(population).__name__}')

        pp = population.population_parameters
        trainable = pp._schema.trainable
        if not trainable:
            raise ValueError('Population has no trainable parameters')

        self.population = population
        self.opt_cls = opt_cls
        self._opt_args = tuple(opt_args)
        self.name = default(name, getattr(opt_cls, '__name__', 'opt').lower())
        self._share_all, self._shared_keys = parse_shared_keys(shared)
        self._shared_reduce = shared_reduce

        sig_params, self._accepted_kwargs = inspect_optimizer_signature(opt_cls)
        store = pp.register_opt_store(self.name)
        named = pp.named_parameters_dict()
        coerced, probe_kw = register_hyperparams(store, pp.pop_size, opt_kwargs, sig_params, named[trainable[0].name])

        defaults, probe_state, self._can_batch, self._can_batch_hetero = probe_optimizer(
            opt_cls, self._opt_args, probe_kw, trainable, named, self._accepted_kwargs,
            test_batch = not per_individual,
        )

        self._alloc_from_probe(store, probe_state)
        defaults.update(coerced)
        pp._optimizers[self.name] = self
        pp._default_opt_name = self.name

        super().__init__(self._leaf_params(), defaults)
        self._sync_optimizer_state()

    # properties & allocation

    @property
    def _pp(self) -> PopulationParameters:
        return self.population.population_parameters

    @property
    def _store(self) -> OptStore:
        return self._pp.opt_stores[self.name]

    @property
    def shared_keys(self) -> frozenset[str]:
        return frozenset(self._store.state_keys()) if self._share_all else frozenset(self._shared_keys)

    @property
    def is_shared(self) -> bool:
        keys = set(self._store.state_keys())
        return self._share_all or (bool(keys) and keys.issubset(self._shared_keys))

    def _is_key_shared(self, key: str) -> bool:
        return self._share_all or key in self._shared_keys

    def _alloc_from_probe(self, store: OptStore, probe_state: dict, cpu_keys: set[tuple[str, str]] | None = None):
        pp = self._pp
        trainable = pp._schema.trainable
        named = pp.named_parameters_dict()
        device = named[trainable[0].name].device
        init_shape = (pp.pop_size, len(trainable)) if pp.is_population else (len(trainable),)
        store.alloc_global('_init', torch.zeros(init_shape, device = device, dtype = torch.bool), populated = pp.is_population)

        slots: dict[str, dict[str, torch.dtype]] = {}
        for spec in trainable:
            for key, val in probe_state.get(spec.name, {}).items():
                if tuple(val.shape) == spec.shape:
                    slots.setdefault(key, {})[spec.name] = val.dtype
                    continue
                is_cpu = (key, spec.name) in cpu_keys if exists(cpu_keys) else (val.ndim == 0 and val.device.type == 'cpu')
                store.alloc_aux(
                    key, spec.name, tuple(val.shape),
                    dtype = val.dtype,
                    device = torch.device('cpu') if is_cpu else named[spec.name].device,
                    shared = self._is_key_shared(key),
                    cpu_scalar = is_cpu,
                )

        for key, dtypes in slots.items():
            store.alloc_slot(key, list(dtypes), shared = self._is_key_shared(key), dtypes = dtypes)

    def _write_state_(self, store: OptStore, spec: TensorSpec, key: str, value: Tensor, idx: int | None, views: dict):
        is_shared = self._is_key_shared(key)
        if tuple(value.shape) == spec.shape:
            if key not in store.slots:
                store.alloc_slot(key, shared = is_shared)
                views[key] = store.slot_views(key)
            buf = views[key].get(spec.name)
        else:
            if spec.name not in store.aux.get(key, ()):
                store.alloc_aux(key, spec.name, tuple(value.shape), dtype = value.dtype, device = value.device, shared = is_shared)
            buf = store.get(f'aux/{key}/{spec.name}')

        if not exists(buf):
            return
        dst = buf[idx] if (exists(idx) and buf.ndim == value.ndim + 1) else buf
        if value is not dst:
            dst.copy_(value.detach().to(dst))

    # hyperparams & Optimizer sync

    def _sync_hp_from_groups(self):
        store = self._store
        for key, val in self.param_groups[0].items():
            buf = store.get(f'hp/{key}')
            if exists(buf) and is_tensor(val) and val is not buf and val.shape == buf.shape:
                with torch.no_grad():
                    buf.copy_(val.detach().to(buf))

    def _resolve_kwargs(self, *, index = None, collapse: bool = False) -> dict:
        self._sync_hp_from_groups()
        store = self._store
        out = {}
        for key, val in self.param_groups[0].items():
            if key == 'params':
                continue
            buf = store.get(f'hp/{key}')
            if not exists(buf):
                out[key] = val
            elif collapse:
                out[key] = float((buf[index] if exists(index) else buf).float().mean())
            elif exists(index):
                out[key] = buf[index].item() if isinstance(index, int) else buf[index]
            else:
                out[key] = buf
        return out

    def _filter_kwargs(self, kwargs: dict, *, differentiable: bool = False) -> dict:
        return filter_optimizer_kwargs(kwargs, self._accepted_kwargs, differentiable = differentiable)

    def _leaf_params(self) -> list[Tensor]:
        pp = self._pp
        if pp.is_population:
            return list(pp._leaves)
        named = pp.named_parameters_dict()
        return [named[s.name] for s in pp._schema.trainable]

    def _sync_optimizer_state(self):
        group = self.param_groups[0]
        group['params'] = self._leaf_params()
        for key, buf in self._store.items('hp/'):
            if key in group:
                group[key] = buf

    # state views, extract & writeback

    @staticmethod
    def _param_state(
        store: OptStore,
        spec: TensorSpec,
        idx: int | None,
        views: dict[str, dict[str, Tensor]],
        *,
        clone: bool = False,
        detach: bool = False,
        clone_shared: bool = False,
        to_cpu: bool = False,
    ) -> dict[str, Tensor]:
        slice_member = store._pp.is_population and exists(idx)
        state: dict[str, Tensor] = {}

        for key, slot_views in views.items():
            if spec.name not in slot_views:
                continue
            val = slot_views[spec.name]
            if slice_member and val.ndim > len(spec.shape):
                val = val[idx]
            if clone or (clone_shared and store.slots[key].shared and not spec.shared):
                val = val.detach().clone() if detach else val.clone()
            state[key] = val

        for aux_key, params in store.aux.items():
            meta = params.get(spec.name)
            if not exists(meta):
                continue
            buf = store.get(f'aux/{aux_key}/{spec.name}')
            if not exists(buf):
                continue
            val = buf[idx] if (slice_member and not meta.shared) else buf
            if clone or to_cpu:
                val = val.detach().clone() if detach else val.clone()
            state[aux_key] = val.cpu() if (to_cpu and meta.cpu_scalar) else val

        return state

    def named_state(self, pp: PopulationParameters, *, idx: int | None = None, clone: bool = False) -> dict[str, dict[str, Tensor]]:
        store = pp.opt_stores[self.name]
        views = store.all_views()
        return {s.name: self._param_state(store, s, idx, views, clone = clone, detach = True) for s in pp._schema.trainable}

    def state_dict_for(self, pp: PopulationParameters, *, idx: int | None = None, clone: bool = False) -> dict:
        store = pp.opt_stores[self.name]
        named_opt = self.named_state(pp, idx = idx, clone = clone)
        trainable = pp._schema.trainable
        group = self._resolve_kwargs(index = idx)
        group['params'] = list(range(len(trainable)))
        state = {i: ({} if (exists(idx) and not store.is_init(s, idx)) else named_opt[s.name]) for i, s in enumerate(trainable)}
        return {'state': state, 'param_groups': [group]}

    @torch.no_grad()
    def load_individual_state_dict_(self, pp: PopulationParameters, idx: int, data: Mapping):
        store = pp.opt_stores[self.name]
        named_src, opt_sd = data.get('named_optimizer_state'), data.get('optimizer_state')
        idx_src = opt_sd.get('state', {}) if exists(opt_sd) else {}
        opt_idx = idx if pp.is_population else None
        views = store.all_views()

        for i, spec in enumerate(pp._schema.trainable):
            entry = named_src.get(spec.name) if exists(named_src) else idx_src.get(i)
            has_state = bool(entry)
            if has_state:
                for key, val in entry.items():
                    if is_tensor(val):
                        self._write_state_(store, spec, key, val, opt_idx, views)
            elif pp.is_population and not spec.shared:
                for key, tensor in self._param_state(store, spec, opt_idx, views).items():
                    if not self._is_key_shared(key):
                        tensor.zero_()

            if not spec.shared or has_state or not pp.is_population:
                store.mark_init_([spec], None if (spec.shared or not pp.is_population) else [opt_idx], value = has_state)

        if pp.is_population and exists(opt_sd) and opt_sd.get('param_groups'):
            copy_hyperparams_(store, opt_sd['param_groups'][0], [idx])

    def slot(self, name: str) -> ParamSlot:
        pp = self._pp
        spec = pp._schema.by_name.get(name)
        if not exists(spec) or not spec.requires_grad:
            raise KeyError(f'no tracked trainable parameter named {name!r}')
        return ParamSlot(name, pp.named_parameters_dict()[name], pp.named_grads_dict()[name], self.named_state(pp)[name])

    def extract(self, index: int = 0) -> tuple[Module, Optimizer]:
        pp, store = self._pp, self._store
        idx = pp._normalize_index(index)
        ind = self.population[idx]
        opt_idx = idx if pp.is_population else None

        kwargs = self._filter_kwargs(self._resolve_kwargs(index = opt_idx))
        ind_opt = self.opt_cls([p for p in ind.parameters() if p.requires_grad], *self._opt_args, **kwargs)

        ind_named, views = dict(ind.named_parameters()), store.all_views()
        for spec in pp._schema.trainable:
            if not store.is_init(spec, opt_idx):
                continue
            state = self._param_state(store, spec, opt_idx, views, clone = True, to_cpu = True)
            if state:
                ind_opt.state[ind_named[spec.name]] = state
        return ind, ind_opt

    def load_individual_state_(self, index: int, module: Module, optimizer: Optimizer):
        pp, named = self._pp, dict(module.named_parameters())
        names = [s.name for s in pp._schema.trainable]
        if any(name not in named for name in names):
            raise ValueError('source module is missing trainable parameters')

        payload: dict = {'named_optimizer_state': {n: st for n in names if (st := optimizer.state.get(named[n]))}}
        if optimizer.param_groups:
            payload['optimizer_state'] = {'param_groups': optimizer.param_groups}
        self.load_individual_state_dict_(pp, pp._normalize_index(index), payload)

    # share / unshare, reset & zero_grad

    def _reduce_shared(self, key: str, tensor: Tensor, reduction = None) -> Tensor:
        spec = default(reduction, self._shared_reduce)
        if isinstance(spec, Mapping):
            spec = spec.get(key, 'mean')
        return reduce_pop(tensor, spec)

    @torch.no_grad()
    def _convert_sharing(self, keys, *, to_shared: bool, reduce = None):
        share_all, target = parse_shared_keys(keys)
        pp, store = self._pp, self._store
        affected = set(store.state_keys()) if share_all else target

        if pp.is_population:
            def resize(key: str, t: Tensor) -> Tensor:
                return self._reduce_shared(key, t, reduce).clone() if to_shared else repeat(t, '... -> p ...', p = pp.pop_size).clone()

            for key, slot in store.slots.items():
                if key in affected and slot.shared != to_shared:
                    slot.shared = to_shared
                    if slot.pop_names:
                        store.set(f'slot/{key}/pop', resize(key, store[f'slot/{key}/pop']), populated = not to_shared)

            for aux_key, params in store.aux.items():
                if aux_key not in affected:
                    continue
                for param_name, meta in params.items():
                    if not pp.is_shared_param(param_name) and meta.shared != to_shared:
                        meta.shared = to_shared
                        buf_key = f'aux/{aux_key}/{param_name}'
                        store.set(buf_key, resize(aux_key, store[buf_key]), populated = not to_shared)

            flags = store.global_tensor('_init')
            if to_shared and exists(flags) and flags.ndim == 2:
                flags.copy_(flags.any(dim = 0, keepdim = True).expand_as(flags))

        if share_all:
            self._share_all, self._shared_keys = to_shared, set()
        else:
            current = set(store.state_keys()) if self._share_all else self._shared_keys
            self._shared_keys = (current | target) if to_shared else (current - target)
            self._share_all &= to_shared

        self._sync_optimizer_state()
        return self

    def share(self, keys: bool | str | Iterable[str] = True, *, reduce = None):
        return self._convert_sharing(keys, to_shared = True, reduce = reduce)

    def unshare(self, keys: bool | str | Iterable[str] = True):
        return self._convert_sharing(keys, to_shared = False)

    def _member_indices(self, members) -> list[int] | None:
        if not exists(members):
            return None
        indices = list(dict.fromkeys(self._pp.resolve_members(members, non_empty = True)))
        return indices if self._pp.is_population else None

    def reset(self, members = None):
        self._store.reset_members_(self._member_indices(members))
        return self

    def zero_grad(self, set_to_none: bool = True, *, members = None):
        self._pp.zero_grad_(members = members, set_to_none = set_to_none)

    # stepping (batched fast path + per-member general path)

    def _run_inner(self, params: list[Tensor], grads: list[Tensor], states: list[dict], kwargs: dict) -> list[dict]:
        inner = self.opt_cls(params, *self._opt_args, **kwargs)
        for param, grad, state in zip(params, grads, states):
            param.grad = grad
            if state:
                inner.state[param] = state
        inner.step()
        return [inner.state.get(param, {}) for param in params]

    def _try_batched_step(
        self,
        pp: PopulationParameters,
        store: OptStore,
        flat_param: Tensor | None,
        grads,
        *,
        inplace: bool,
        active_indices: list[int] | None = None,
        create_graph: bool = False,
    ) -> Tensor | None:
        if not (self._can_batch and pp.is_population and exists(flat_param)):
            return None
        if exists(active_indices) or isinstance(grads, Mapping) or store.items('hp/'):
            return None

        specs = pp._schema.pop_trainable
        if not exists(grads) and exists(pp._active_grads) and any(s.name not in pp._active_grads for s in specs):
            return None

        analysis = analyze_batched_state(store, specs, allow_hetero = self._can_batch_hetero)
        if not exists(analysis):
            return None

        mode, is_init = analysis
        flat_grad = grads if exists(grads) else flat_param.grad
        if not exists(flat_grad):
            return flat_param

        wp = work_tensor(flat_param, inplace, create_graph)
        wg = flat_grad.detach() if inplace else flat_grad
        first = specs[0].name
        state: dict[str, Tensor] = {}

        if is_init:
            for key in store.slots:
                buf = store[f'slot/{key}/pop']
                state[key] = (
                    repeat(buf, 'd -> p d', p = pp.pop_size).clone()
                    if buf.ndim == 1
                    else (buf if inplace else buf.clone())
                )

            for aux_key in store.aux_keys_for(first):
                meta = store.aux[aux_key][first]
                buf = store[f'aux/{aux_key}/{first}']
                if mode == 'hetero':
                    val = buf.clone() if meta.shared else buf.unsqueeze(-1).clone()
                    state[aux_key] = val.to(wp.device)
                else:
                    val = (buf if meta.shared else buf[0]).clone()
                    state[aux_key] = val.cpu() if meta.cpu_scalar else val

        kwargs = self._filter_kwargs(self._resolve_kwargs(), differentiable = create_graph or mode == 'hetero')
        next_state = self._run_inner([wp], [wg], [state], kwargs)[0]

        for key, val in next_state.items():
            if not is_tensor(val):
                continue
            if key in store.slots:
                slot_key = f'slot/{key}/pop'
                is_pop = store[slot_key].ndim == 2
                store.put(slot_key, val if is_pop else self._reduce_shared(key, val), populated = is_pop, inplace = inplace)
            elif first in store.aux.get(key, ()):
                is_shared = store.aux[key][first].shared
                val_out = val.squeeze(-1) if (not is_shared and val.ndim == 2 and val.shape[-1] == 1) else val
                for spec in specs:
                    aux_buf_key = f'aux/{key}/{spec.name}'
                    val_cast = val_out.to(store[aux_buf_key])
                    if not is_shared:
                        val_cast = val_cast.expand_as(store[aux_buf_key])
                    if not inplace:
                        val_cast = val_cast.clone()
                    store.put(aux_buf_key, val_cast, populated = not is_shared, inplace = inplace)

        store.mark_init_(specs)
        return wp

    def _commit_group_states_(
        self,
        store: OptStore,
        items: list[tuple[TensorSpec, Tensor, Tensor]],
        indices: list[int] | None,
        stepped_states: list[list[dict]],
        views: dict[str, dict[str, Tensor]],
        *,
        inplace: bool,
    ):
        next_views = None if inplace else {k: dict(v) for k, v in views.items()}

        for (spec, _, _), param_states in zip(items, zip(*stepped_states)):
            state_keys = list(dict.fromkeys(k for st in param_states for k, v in st.items() if is_tensor(v)))
            for key in state_keys:
                vals = [st[key] for st in param_states]
                is_shared_key = exists(indices) and not spec.shared and self._is_key_shared(key)

                if inplace:
                    if is_shared_key:
                        self._write_state_(store, spec, key, self._reduce_shared(key, torch.stack(vals)), None, views)
                    else:
                        for member, val in zip([None] if not exists(indices) else indices, vals):
                            self._write_state_(store, spec, key, val, member, views)
                    continue

                val = vals[0] if not exists(indices) else (
                    self._reduce_shared(key, torch.stack(vals)) if is_shared_key else torch.stack(vals)
                )
                if key in store.slots:
                    next_views[key][spec.name] = val
                elif spec.name in store.aux.get(key, ()):
                    aux_buf_key = f'aux/{key}/{spec.name}'
                    is_pop_aux = exists(indices) and not spec.shared and not is_shared_key
                    store.set(aux_buf_key, val.to(store[aux_buf_key].device), populated = is_pop_aux)

        if not inplace:
            store.pack_slots_(next_views, is_pop_group = exists(indices))

    def _step_group(
        self,
        store: OptStore,
        items: list[tuple[TensorSpec, Tensor, Tensor]],
        indices: list[int] | None,
        *,
        inplace: bool,
        create_graph: bool = False,
        fixed_kwargs: dict | None = None,
    ) -> list[list[Tensor]]:
        views = store.all_views()
        loop = [(0, None)] if not exists(indices) else list(enumerate(indices))
        stepped_params, stepped_states = [], []

        for pos, member in loop:
            kwargs = fixed_kwargs if exists(fixed_kwargs) else self._filter_kwargs(
                self._resolve_kwargs(index = member), differentiable = create_graph,
            )
            step_p = [work_tensor(p if not exists(member) else p[member], inplace, create_graph) for _, p, _ in items]
            step_g = [g if not exists(member) else g[pos] for _, _, g in items]
            if inplace:
                step_g = [g.detach() for g in step_g]
            step_st = [
                self._param_state(store, s, member, views, clone = not inplace, clone_shared = exists(indices), to_cpu = True)
                if store.is_init(s, member) else {}
                for s, _, _ in items
            ]
            stepped_states.append(self._run_inner(step_p, step_g, step_st, kwargs))
            stepped_params.append(step_p)

        store.mark_init_([s for s, _, _ in items], indices)
        self._commit_group_states_(store, items, indices, stepped_states, views, inplace = inplace)
        return stepped_params

    def step(self, closure = None, *, grads: Mapping[str, Tensor] | Tensor | None = None, members = None):
        loss = None
        if exists(closure):
            with torch.enable_grad():
                loss = closure()

        pp, store = self._pp, self._store
        schema = pp._schema
        active_indices = self._member_indices(members)

        with torch.no_grad():
            pop_done = False
            if pp.is_population and schema.pop_trainable:
                pop_done = exists(self._try_batched_step(pp, store, pp._pop_leaf, grads, inplace = True, active_indices = active_indices))
                if pop_done and not schema.shared_trainable:
                    return loss

            named_p = pp.named_parameters_dict()
            named_g = pp.unpack_flat(grads) if is_tensor(grads) else (grads if exists(grads) else pp.named_grads_dict())

            shared_specs = schema.shared_trainable if pp.is_population else schema.trainable
            groups: list[tuple[list[TensorSpec], list[int] | None]] = [(shared_specs, None)]
            if pp.is_population and schema.pop_trainable and not pop_done:
                groups.append((schema.pop_trainable, default(active_indices, list(range(pp.pop_size)))))

            for specs, group_indices in groups:
                items = [
                    (s, named_p[s.name], match_grad(s, named_g[s.name], active_indices, pp.pop_size))
                    for s in specs if exists(named_g.get(s.name))
                ]
                if not items:
                    continue
                fixed_kwargs = None if exists(group_indices) else self._filter_kwargs(
                    self._resolve_kwargs(index = active_indices, collapse = pp.is_population)
                )
                self._step_group(store, items, group_indices, inplace = True, fixed_kwargs = fixed_kwargs)

        return loss

    def functional_step(
        self,
        loss: Tensor | None = None,
        params: PopulationParameters | None = None,
        *,
        grads: Mapping[str, Tensor] | Tensor | None = None,
        create_graph: bool = True,
        retain_graph: bool | None = None,
        detach_preconditioner: bool = True,
    ) -> PopulationParameters:
        if exists(loss) == exists(grads):
            raise ValueError('pass exactly one of loss or grads to functional_step')

        pp = default(params, self._pp)
        if not pp.is_population:
            raise ValueError('functional_step requires a populated PopulationParameters')
        if self.name not in pp.opt_stores:
            raise ValueError(f'params has no optimizer state for {self.name!r} (was it cloned with include_opt=False?)')

        pop_leaf, shared_leaf = pp._pop_leaf, pp._shared_leaf
        pop_grad: Tensor | None = None
        shared_grad: Tensor | None = None
        named_grads: Mapping[str, Tensor] | None = None

        if exists(grads):
            if is_tensor(grads):
                pop_grad = grads
            elif isinstance(grads, Mapping):
                named_grads = grads
                pop_specs = pp._schema.pop_trainable
                if exists(pop_leaf) and pop_specs and all(exists(grads.get(s.name)) for s in pop_specs):
                    pop_grad, _ = pack([match_grad(s, grads[s.name], None, pp.pop_size) for s in pop_specs], 'p *')
            else:
                raise TypeError(f'grads must be Tensor or Mapping, got {type(grads).__name__}')
        else:
            targets = [t for t in (pop_leaf, shared_leaf) if exists(t)]
            raw_grads = torch_grad(
                loss, targets,
                only_inputs = True,
                create_graph = create_graph,
                retain_graph = default(retain_graph, create_graph),
                allow_unused = True,
            ) if targets else ()
            grad_by_id = {id(t): g for t, g in zip(targets, raw_grads)}
            pop_grad = grad_by_id.get(id(pop_leaf)) if exists(pop_leaf) else None
            shared_grad = grad_by_id.get(id(shared_leaf)) if exists(shared_leaf) else None

        if not create_graph:
            pop_grad = pop_grad.detach() if exists(pop_grad) else None
            shared_grad = shared_grad.detach() if exists(shared_grad) else None
            if exists(named_grads):
                named_grads = {k: v.detach() for k, v in named_grads.items() if exists(v)}

        next_store_bank = pp.store.clone(
            detach = True,
            prefixes = ('frozen/', 'buffer/', 'opt/'),
            shallow_prefixes = ('opt/',) if create_graph else (),
        )
        next_pp = pp._spawn(next_store_bank, None, None, functional = create_graph)
        next_store = next_pp.opt_stores[self.name]
        next_store.set('global/_init', next_store['global/_init'].clone(), populated = True)

        grad_ctx = torch.enable_grad() if create_graph else torch.no_grad()
        mode_ctx = _DifferentiableStepMode(detach_preconditioner) if create_graph else nullcontext()

        with grad_ctx, mode_ctx:
            next_pop = pop_leaf
            if exists(pop_leaf) and exists(pop_grad):
                next_pop = self._try_batched_step(
                    next_pp, next_store, pop_leaf, pop_grad,
                    inplace = False, create_graph = create_graph,
                )
                if not exists(next_pop):
                    items = [(s, slice_spec(s, pop_leaf), slice_spec(s, pop_grad)) for s in pp._schema.pop_trainable]
                    stepped = self._step_group(
                        next_store, items, list(range(pp.pop_size)),
                        inplace = False, create_graph = create_graph,
                    )
                    next_pop, _ = pack([torch.stack(ts) for ts in zip(*stepped)], 'p *')
            elif exists(pop_leaf) and exists(named_grads):
                items = [
                    (s, slice_spec(s, pop_leaf), match_grad(s, named_grads[s.name], None, pp.pop_size))
                    for s in pp._schema.pop_trainable if exists(named_grads.get(s.name))
                ]
                if items:
                    stepped = self._step_group(
                        next_store, items, list(range(pp.pop_size)),
                        inplace = False, create_graph = create_graph,
                    )
                    updated = {s.name: torch.stack(ts) for (s, _, _), ts in zip(items, zip(*stepped))}
                    next_pop, _ = pack([updated.get(s.name, slice_spec(s, pop_leaf)) for s in pp._schema.pop_trainable], 'p *')

            next_shared = shared_leaf
            if exists(shared_leaf) and (exists(shared_grad) or exists(named_grads)):
                if exists(shared_grad):
                    items = [(s, slice_spec(s, shared_leaf), slice_spec(s, shared_grad)) for s in pp._schema.shared_trainable]
                else:
                    items = [
                        (s, slice_spec(s, shared_leaf), match_grad(s, named_grads[s.name], None, pp.pop_size))
                        for s in pp._schema.shared_trainable if exists(named_grads.get(s.name))
                    ]
                if items:
                    fixed_kwargs = self._filter_kwargs(self._resolve_kwargs(collapse = True), differentiable = create_graph)
                    stepped = self._step_group(
                        next_store, items, None,
                        inplace = False, create_graph = create_graph, fixed_kwargs = fixed_kwargs,
                    )
                    updated = {s.name: t for (s, _, _), t in zip(items, stepped[0])}
                    next_shared, _ = pack([updated.get(s.name, slice_spec(s, shared_leaf)) for s in pp._schema.shared_trainable], '*')

        if create_graph:
            next_pp._func_leaves = (next_pop, next_shared)
        else:
            def to_param(cur: Tensor | None, orig: Tensor | None) -> Parameter | None:
                if not exists(cur):
                    return None
                return Parameter(cur.detach().clone() if cur is orig else cur.detach(), requires_grad = True)

            next_pp._pop_param = to_param(next_pop, pop_leaf)
            next_pp._shared_param = to_param(next_shared, shared_leaf)
        return next_pp

    # lifecycle hooks

    @torch.no_grad()
    def _on_populate(self, pop_size: int, *, init: Literal['reinit', 'clone'] | object, preserve_first: bool):
        pp, store = self._pp, self._store

        def expand(t: Tensor) -> Tensor:
            if init == 'clone':
                return repeat(t, '... -> p ...', p = pop_size).clone()
            out = torch.zeros((pop_size, *t.shape), device = t.device, dtype = t.dtype)
            if preserve_first:
                out[0].copy_(t)
            return out

        old_state = self.named_state(pp, clone = True)
        old_init = store['global/_init'].clone()
        cpu_keys = {(k, pname) for k, params in store.aux.items() for pname, meta in params.items() if meta.cpu_scalar}

        store.clear(keep_hp = True)
        self._alloc_from_probe(store, old_state, cpu_keys)

        views, flags = store.all_views(), store['global/_init']
        for spec in pp._schema.trainable:
            for key, val in old_state[spec.name].items():
                buf = views[key][spec.name] if key in store.slots else store[f'aux/{key}/{spec.name}']
                buf.copy_(expand(val) if buf.ndim == val.ndim + 1 else val)
            flag = old_init[spec.trainable_index]
            flags[:, spec.trainable_index] = flag if spec.shared else expand(flag)

        self._sync_optimizer_state()

    def _clone_for_subpopulation(self, target_pop: Population, indices: list[int]) -> PopulationOptimizer:
        is_pop = self.population.is_population
        kwargs = self._resolve_kwargs(index = indices if is_pop else None)
        dst_pp = target_pop.population_parameters

        sub = PopulationOptimizer.__new__(PopulationOptimizer)
        sub.population = target_pop
        for attr in ('opt_cls', '_opt_args', 'name', '_share_all', '_shared_reduce', '_accepted_kwargs', '_can_batch', '_can_batch_hetero'):
            setattr(sub, attr, getattr(self, attr))
        sub._shared_keys = set(self._shared_keys)
        dst_pp._optimizers[self.name] = sub

        Optimizer.__init__(sub, sub._leaf_params(), kwargs)
        if is_pop:
            sub._sync_optimizer_state()
        else:
            sub_store = self._store.clone_handle(dst_pp)
            dst_pp.opt_stores[self.name] = sub_store
            for key, tensor in self._store.items():
                sub_store.set(key, tensor.detach().clone(), populated = False)
            sub._on_populate(len(indices), init = 'clone', preserve_first = True)
        return sub

    @torch.no_grad()
    def _copy_from_other_opt_(self, dst_indices: list[int], src_opt: PopulationOptimizer, src_indices: list[int]):
        if [s.name for s in self._pp._schema.trainable] != [s.name for s in src_opt._pp._schema.trainable]:
            raise ValueError('source optimizer parameter names do not match target optimizer')

        for key, dst in self._store.items():
            if key.startswith('hp/') or key not in src_opt._store:
                continue
            src = src_opt._store[key]
            src_is_pop = src_opt._store.is_populated(key, src_opt._pp.pop_size)
            if self._store.is_populated(key, self._pp.pop_size):
                dst[dst_indices] = (src[src_indices] if src_is_pop else src).to(dst)
            else:
                dst.copy_((src[src_indices[-1]] if src_is_pop else src).to(dst))

        if self._pp.is_population:
            hp_src_indices = src_indices if src_opt._pp.is_population else None
            copy_hyperparams_(self._store, src_opt._resolve_kwargs(), dst_indices, hp_src_indices)
