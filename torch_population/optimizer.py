from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable, Mapping
from typing import TYPE_CHECKING, Literal

import torch
from einops import pack, repeat
from torch import Tensor, is_tensor
from torch.autograd import grad as torch_grad
from torch.nn import Module
from torch.optim import Optimizer

from torch_population.parameters import PopulationParameters
from torch_population.store import (
    OptStore,
    ParamSlot,
    TensorSpec,
    default,
    exists,
    is_sequence,
    reduce_mean,
    slice_spec,
)

if TYPE_CHECKING:
    from torch_population.torch_population import Population

# helpers

def _parse_shared(shared) -> tuple[bool, set[str]]:
    if isinstance(shared, bool):
        return shared, set()
    if isinstance(shared, str):
        return False, {shared}
    items = list(shared)
    if not all(isinstance(k, str) for k in items):
        raise TypeError('shared state keys must be strings')
    return False, set(items)

def _inspect_opt(opt_cls):
    try:
        target = opt_cls.__init__ if isinstance(opt_cls, type) else opt_cls
        params = inspect.signature(target).parameters
    except (TypeError, ValueError):
        return {}, None
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return params, None
    return params, {k for k, p in params.items() if p.kind.name in ('POSITIONAL_OR_KEYWORD', 'KEYWORD_ONLY')}

def _work_tensor(t: Tensor, inplace: bool, create_graph: bool) -> Tensor:
    return t.detach() if inplace else (t.clone() if create_graph else t.detach().clone())

def _put_buf(store: OptStore, key: str, val: Tensor, populated: bool, inplace: bool):
    if not inplace:
        store.set(key, val, populated = populated)
    elif val is not store[key]:
        store[key].copy_(val)

def _match_grad(spec: TensorSpec, grad: Tensor, active_idx: list[int] | None, pop_size: int | None) -> Tensor:
    if not exists(pop_size):
        return grad
    if spec.shared:
        return grad if grad.ndim == len(spec.shape) else (grad[active_idx] if exists(active_idx) else grad).sum(0)
    if not exists(active_idx):
        return grad
    if grad.ndim == len(spec.shape) + 1 and grad.shape[0] in (pop_size, len(active_idx)):
        return grad[active_idx] if grad.shape[0] == pop_size else grad
    if len(active_idx) == 1 and tuple(grad.shape) == spec.shape:
        return grad.unsqueeze(0)
    raise ValueError(f'grad shape {tuple(grad.shape)} incompatible with {spec.name!r} {spec.shape}')

# hyperparameters (scalar or per-individual 1D tensor)

def _is_num_seq(v) -> bool:
    return is_sequence(v) and bool(v) and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v)

def _register_hp(store: OptStore, pop_size: int | None, opt_kwargs: dict, sig_params: Mapping, ref: Tensor):
    coerced, probe = {}, {}
    for k, v in opt_kwargs.items():
        p = sig_params.get(k)
        if exists(p) and isinstance(p.default, tuple) and is_sequence(v):
            coerced[k] = probe[k] = tuple(v)
            continue
        if not is_tensor(v) and not _is_num_seq(v):
            coerced[k] = probe[k] = v
            continue
        if not is_tensor(v):
            v = torch.tensor([float(x) for x in v], device = ref.device, dtype = ref.dtype)
        t = v.detach().to(device = ref.device, dtype = ref.dtype if v.is_floating_point() else v.dtype)
        if t.ndim == 0 or (not exists(pop_size) and t.shape == (1,)):
            coerced[k] = probe[k] = t.item()
            continue
        if not exists(pop_size) or t.shape != (pop_size,):
            raise ValueError(f'per-individual {k} requires shape ({pop_size},), got {tuple(t.shape)}')
        coerced[k] = store.set(f'hp/{k}', t.clone(), populated = True, append_repeat = True)
        probe[k] = coerced[k][0].item()
    return coerced, probe

def _copy_hp_(dst_store: OptStore, src_kwargs: dict, dst_idx: list[int], src_idx = None):
    for k, dst in dst_store.items('hp/'):
        if k not in src_kwargs or not exists(val := src_kwargs[k]):
            continue
        if is_tensor(val) and exists(src_idx):
            dst[dst_idx] = val[src_idx].to(dst)
        else:
            dst[dst_idx] = val.to(dst) if is_tensor(val) else float(val)

# optimizer probing & batch uniformity

def _probe_can_batch(opt_cls, opt_args: tuple, opt_kwargs: dict, ref: Tensor) -> bool:
    try:
        p0, p1 = torch.randn(2, 3, 2, device = ref.device, dtype = ref.dtype)
        flat = torch.stack((p0.flatten(), p1.flatten()))
        params = (flat, p0.clone(), p1.clone())
        opts = [opt_cls([p], *opt_args, **opt_kwargs) for p in params]
        for _ in range(2):
            g0, g1 = torch.randn(2, 3, 2, device = ref.device, dtype = ref.dtype)
            for p, g, o in zip(params, (torch.stack((g0.flatten(), g1.flatten())), g0, g1), opts):
                p.grad = g
                o.step()
        shapes_ok = all(not is_tensor(v) or v.shape in (flat.shape, ()) for v in opts[0].state.get(flat, {}).values())
        return shapes_ok and torch.allclose(flat.view(2, 3, 2), torch.stack(params[1:]), atol = 1e-5, rtol = 1e-4)
    except Exception:
        return False

@torch.no_grad()
def _probe_optimizer(opt_cls, opt_args, opt_kwargs, specs, named, *, test_batch: bool):
    with torch.random.fork_rng():
        torch.manual_seed(0)
        dummies = [torch.zeros(s.shape, device = named[s.name].device, dtype = named[s.name].dtype) for s in specs]
        for d in dummies:
            d.grad = torch.ones_like(d)
        probe = opt_cls(dummies, *opt_args, **opt_kwargs)
        probe.step()
        probe.step()
        defaults = {k: v for k, v in probe.param_groups[0].items() if k != 'params'}
        state = {s.name: {k: v for k, v in probe.state.get(d, {}).items() if is_tensor(v)} for s, d in zip(specs, dummies)}
        can_batch = _probe_can_batch(opt_cls, opt_args, opt_kwargs, named[specs[0].name]) if test_batch else False
    return defaults, state, can_batch

def _uniform_init(store: OptStore, specs: list[TensorSpec]) -> bool | None:
    names = [s.name for s in specs]
    if any(slot.pop_names != names for slot in store.slots.values()):
        return None
    flags = store['global/_init'][:, [s.trainable_index for s in specs]]
    if not (flags == flags[0, 0]).all():
        return None
    if not flags[0, 0]:
        return False

    first = specs[0].name
    for ak in {k for k, _ in store.aux}:
        ref = store.get(f'aux/{ak}/{first}')
        if not exists(ref):
            return None
        val = ref if store.aux[(ak, first)].shared else ref[0]
        if val.ndim != 0:
            return None
        for s in specs:
            t = store.get(f'aux/{ak}/{s.name}')
            if not exists(t) or not (t == val).all():
                return None
    return True

# colocated population optimizer

class PopulationOptimizer(Optimizer):
    def __init__(self, population: Population, opt_cls: type[Optimizer] | Callable[..., Optimizer], *opt_args, shared: bool | str | Iterable[str] = False, per_individual: bool = False, name: str | None = None, **opt_kwargs):
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
        self._share_all, self._shared_keys = _parse_shared(shared)

        sig_params, self._accepted_kwargs = _inspect_opt(opt_cls)
        store = pp.register_opt_store(self.name)
        named = pp.named_parameters_dict()
        coerced, probe_kw = _register_hp(store, pp.pop_size, opt_kwargs, sig_params, named[trainable[0].name])
        defaults, probe_state, self._can_batch = _probe_optimizer(opt_cls, self._opt_args, probe_kw, trainable, named, test_batch = not per_individual)

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

    @staticmethod
    def _is_init(store: OptStore, spec: TensorSpec, idx: int | None) -> bool:
        flags = store.global_tensor('_init')
        if not exists(flags):
            return True
        if flags.ndim == 1:
            return bool(flags[spec.trainable_index])
        col = flags[:, spec.trainable_index]
        return bool(col.any() if (spec.shared or not exists(idx)) else col[idx])

    def _alloc_from_probe(self, store: OptStore, probe_state: dict, cpu_keys: set[tuple[str, str]] | None = None):
        pp = self._pp
        trainable = pp._schema.trainable
        named = pp.named_parameters_dict()
        device = named[trainable[0].name].device
        init_shape = (pp.pop_size, len(trainable)) if pp.is_population else (len(trainable),)
        store.alloc_global('_init', torch.zeros(init_shape, device = device, dtype = torch.bool), populated = pp.is_population)

        slots: dict[str, dict[str, torch.dtype]] = {}
        for s in trainable:
            for k, v in probe_state.get(s.name, {}).items():
                if tuple(v.shape) == s.shape:
                    slots.setdefault(k, {})[s.name] = v.dtype
                    continue
                is_cpu = (k, s.name) in cpu_keys if exists(cpu_keys) else (v.ndim == 0 and v.device.type == 'cpu')
                store.alloc_aux(
                    k, s.name, tuple(v.shape), dtype = v.dtype,
                    device = torch.device('cpu') if is_cpu else named[s.name].device,
                    shared = self._is_key_shared(k), cpu_scalar = is_cpu,
                )
        for k, dtypes in slots.items():
            store.alloc_slot(k, list(dtypes), shared = self._is_key_shared(k), dtypes = dtypes)

    def _write_state_(self, store: OptStore, spec: TensorSpec, key: str, value: Tensor, idx: int | None, views: dict):
        is_sh = self._is_key_shared(key)
        if tuple(value.shape) == spec.shape:
            if key not in store.slots:
                store.alloc_slot(key, shared = is_sh)
                views[key] = store.slot_views(key)
            buf = views[key].get(spec.name)
        else:
            if (key, spec.name) not in store.aux:
                is_cpu = value.ndim == 0 and value.device.type == 'cpu'
                store.alloc_aux(key, spec.name, tuple(value.shape), dtype = value.dtype, device = value.device, shared = is_sh, cpu_scalar = is_cpu)
            buf = store.get(f'aux/{key}/{spec.name}')

        if not exists(buf):
            return
        dst = buf[idx] if (exists(idx) and buf.ndim == value.ndim + 1) else buf
        if value is not dst:
            dst.copy_(value.detach().to(dst))

    # hyperparams & Optimizer sync

    def _resolve_kwargs(self, *, index = None, collapse: bool = False) -> dict:
        store = self._store
        out = {}
        for k, v in self.param_groups[0].items():
            if k == 'params':
                continue
            buf = store.get(f'hp/{k}')
            if not exists(buf):
                out[k] = v
                continue
            if is_tensor(v) and v is not buf and v.shape == buf.shape:
                with torch.no_grad():
                    buf.copy_(v.detach().to(buf))
            if collapse:
                out[k] = float((buf[index] if exists(index) else buf).float().mean())
            elif exists(index):
                out[k] = buf[index].item() if isinstance(index, int) else buf[index]
            else:
                out[k] = buf
        return out

    def _filter_kwargs(self, kw: dict, *, differentiable: bool = False) -> dict:
        acc = self._accepted_kwargs
        out = {k: v for k, v in kw.items() if not exists(acc) or k in acc}
        if not differentiable:
            return out
        if not exists(acc) or 'differentiable' in acc:
            out['differentiable'] = True
        for flag in ('foreach', 'fused'):
            if exists(acc) and flag in acc:
                out[flag] = False
        return out

    def _leaf_params(self) -> list[Tensor]:
        pp = self._pp
        if pp.is_population:
            return list(pp._leaves)
        named = pp.named_parameters_dict()
        return [named[s.name] for s in pp._schema.trainable]

    def _sync_optimizer_state(self):
        pg = self.param_groups[0]
        pg['params'] = self._leaf_params()
        for k, buf in self._store.items('hp/'):
            if k in pg:
                pg[k] = buf

    # state views, extract & writeback

    @staticmethod
    def _param_state(store: OptStore, spec: TensorSpec, idx: int | None, views: dict[str, dict[str, Tensor]], *, clone: bool = False, detach: bool = False, clone_shared: bool = False, to_cpu: bool = False) -> dict[str, Tensor]:
        slice_p = store._pp.is_population and exists(idx)
        state: dict[str, Tensor] = {}

        for k, sviews in views.items():
            if spec.name not in sviews:
                continue
            v = sviews[spec.name]
            if slice_p and v.ndim > len(spec.shape):
                v = v[idx]
            if clone or (clone_shared and store.slots[k].shared and not spec.shared):
                v = v.detach().clone() if detach else v.clone()
            state[k] = v

        for (ak, pname), meta in store.aux.items():
            if pname != spec.name:
                continue
            buf = store.get(f'aux/{ak}/{pname}')
            if not exists(buf):
                continue
            v = buf[idx] if (slice_p and not meta.shared) else buf
            if clone or to_cpu:
                v = v.detach().clone() if detach else v.clone()
            state[ak] = v.cpu() if (to_cpu and meta.cpu_scalar) else v
        return state

    def named_state(self, pp: PopulationParameters, *, idx: int | None = None, clone: bool = False) -> dict[str, dict[str, Tensor]]:
        store = pp.opt_stores[self.name]
        views = store.all_views()
        return {s.name: self._param_state(store, s, idx, views, clone = clone, detach = True) for s in pp._schema.trainable}

    def state_dict_for(self, pp: PopulationParameters, *, idx: int | None = None, clone: bool = False) -> dict:
        store = pp.opt_stores[self.name]
        named_opt = self.named_state(pp, idx = idx, clone = clone)
        trainable = pp._schema.trainable
        pg = self._resolve_kwargs(index = idx)
        pg['params'] = list(range(len(trainable)))
        state = {i: ({} if (exists(idx) and not self._is_init(store, s, idx)) else named_opt[s.name]) for i, s in enumerate(trainable)}
        return {'state': state, 'param_groups': [pg]}

    @torch.no_grad()
    def load_individual_state_dict_(self, pp: PopulationParameters, idx: int, data: Mapping):
        store = pp.opt_stores[self.name]
        named_src, opt_sd = data.get('named_optimizer_state'), data.get('optimizer_state')
        idx_src = opt_sd.get('state', {}) if exists(opt_sd) else {}
        opt_idx = idx if pp.is_population else None
        views, flags = store.all_views(), store['global/_init']

        for i, s in enumerate(pp._schema.trainable):
            entry = named_src.get(s.name) if exists(named_src) else idx_src.get(i)
            has_state = bool(entry)
            if has_state:
                for k, v in entry.items():
                    if is_tensor(v):
                        self._write_state_(store, s, k, v, opt_idx, views)
            elif pp.is_population and not s.shared:
                for k, t in self._param_state(store, s, opt_idx, views).items():
                    if not self._is_key_shared(k):
                        t.zero_()

            if flags.ndim == 1:
                flags[s.trainable_index] = has_state
            elif not s.shared or has_state:
                flags[slice(None) if s.shared else opt_idx, s.trainable_index] = has_state

        if pp.is_population and exists(opt_sd) and opt_sd.get('param_groups'):
            _copy_hp_(store, opt_sd['param_groups'][0], [idx])

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
        kw = self._filter_kwargs(self._resolve_kwargs(index = opt_idx))
        ind_opt = self.opt_cls([p for p in ind.parameters() if p.requires_grad], *self._opt_args, **kw)

        ind_named, views = dict(ind.named_parameters()), store.all_views()
        for s in pp._schema.trainable:
            if not self._is_init(store, s, opt_idx):
                continue
            state = self._param_state(store, s, opt_idx, views, clone = True, to_cpu = True)
            if state:
                ind_opt.state[ind_named[s.name]] = state
        return ind, ind_opt

    def load_individual_state_(self, index: int, module: Module, optimizer: Optimizer):
        pp, named = self._pp, dict(module.named_parameters())
        names = [s.name for s in pp._schema.trainable]
        if any(n not in named for n in names):
            raise ValueError('source module is missing trainable parameters')
        payload: dict = {'named_optimizer_state': {n: st for n in names if (st := optimizer.state.get(named[n]))}}
        if optimizer.param_groups:
            payload['optimizer_state'] = {'param_groups': optimizer.param_groups}
        self.load_individual_state_dict_(pp, pp._normalize_index(index), payload)

    # share / unshare, reset & zero_grad

    @torch.no_grad()
    def _convert_sharing(self, keys, *, to_shared: bool):
        share_all, target = _parse_shared(keys)
        pp, store = self._pp, self._store
        affected = set(store.state_keys()) if share_all else target

        if pp.is_population:
            def resize(t: Tensor) -> Tensor:
                return reduce_mean(t, t.dtype).clone() if to_shared else repeat(t, '... -> p ...', p = pp.pop_size).clone()

            for k, slot in store.slots.items():
                if k in affected and slot.shared != to_shared:
                    slot.shared = to_shared
                    if slot.pop_names:
                        store.set(f'slot/{k}/pop', resize(store[f'slot/{k}/pop']), populated = not to_shared)

            for (ak, pname), meta in store.aux.items():
                if ak in affected and not pp.is_shared_param(pname) and meta.shared != to_shared:
                    meta.shared = to_shared
                    key = f'aux/{ak}/{pname}'
                    store.set(key, resize(store[key]), populated = not to_shared)

        if share_all:
            self._share_all, self._shared_keys = to_shared, set()
        else:
            cur = set(store.state_keys()) if self._share_all else self._shared_keys
            self._shared_keys = (cur | target) if to_shared else (cur - target)
            self._share_all &= to_shared
        self._sync_optimizer_state()
        return self

    def share(self, keys: bool | str | Iterable[str] = True):
        return self._convert_sharing(keys, to_shared = True)

    def unshare(self, keys: bool | str | Iterable[str] = True):
        return self._convert_sharing(keys, to_shared = False)

    def _member_indices(self, members) -> list[int] | None:
        if not exists(members):
            return None
        idx = list(dict.fromkeys(self._pp.resolve_members(members, non_empty = True)))
        return idx if self._pp.is_population else None

    def reset(self, members = None):
        self._store.reset_members_(self._member_indices(members))
        return self

    def zero_grad(self, set_to_none: bool = True, *, members = None):
        self._pp.zero_grad_(members = members, set_to_none = set_to_none)

    # stepping (batched fast path + per-member general path)

    def _run_inner(self, params: list[Tensor], grads: list[Tensor], states: list[dict], kw: dict) -> list[dict]:
        inner = self.opt_cls(params, *self._opt_args, **kw)
        for p, g, st in zip(params, grads, states):
            p.grad = g
            if st:
                inner.state[p] = st
        inner.step()
        return [inner.state.get(p, {}) for p in params]

    def _try_batched_step(self, pp: PopulationParameters, store: OptStore, flat_p: Tensor | None, grads, *, inplace: bool, active_idx: list[int] | None = None, create_graph: bool = False) -> Tensor | None:
        schema = pp._schema
        if not (self._can_batch and pp.is_population and exists(flat_p) and not schema.has_shared):
            return None
        if exists(active_idx) or isinstance(grads, Mapping) or store.items('hp/'):
            return None
        if not exists(grads) and exists(pp._active_grads) and len(pp._active_grads) < len(schema.pop_trainable):
            return None

        specs = schema.pop_trainable
        is_init = _uniform_init(store, specs)
        if is_init is None:
            return None

        flat_g = grads if exists(grads) else flat_p.grad
        if not exists(flat_g):
            return flat_p

        state: dict[str, Tensor] = {}
        first = specs[0].name
        if is_init:
            for k in store.slots:
                buf = store[f'slot/{k}/pop']
                state[k] = repeat(buf, 'd -> p d', p = pp.pop_size).clone() if buf.ndim == 1 else (buf if inplace else buf.clone())
            for (ak, pname), meta in store.aux.items():
                if pname == first:
                    buf = store[f'aux/{ak}/{first}']
                    v = (buf if meta.shared else buf[0]).clone()
                    state[ak] = v.cpu() if meta.cpu_scalar else v

        wp = _work_tensor(flat_p, inplace, create_graph)
        wg = flat_g.detach() if inplace else flat_g
        kw = self._filter_kwargs(self._resolve_kwargs(), differentiable = create_graph)
        next_state = self._run_inner([wp], [wg], [state], kw)[0]

        for k, v in next_state.items():
            if not is_tensor(v):
                continue
            if k in store.slots:
                sk = f'slot/{k}/pop'
                is_pop = store[sk].ndim == 2
                _put_buf(store, sk, v if is_pop else v.mean(0), is_pop, inplace)
            elif (k, first) in store.aux:
                for s in specs:
                    ak = f'aux/{k}/{s.name}'
                    is_sh = store.aux[(k, s.name)].shared
                    vt = v.to(store[ak])
                    _put_buf(store, ak, vt if is_sh else vt.expand_as(store[ak]).clone(), not is_sh, inplace)

        store['global/_init'][:, [s.trainable_index for s in specs]] = True
        return wp if (inplace or create_graph) else wp.requires_grad_(True)

    def _step_group(self, store: OptStore, items: list[tuple[TensorSpec, Tensor, Tensor]], indices: list[int] | None, *, inplace: bool, create_graph: bool = False, fixed_kw: dict | None = None) -> list[list[Tensor]]:
        views = store.all_views()
        loop = [(0, None)] if not exists(indices) else list(enumerate(indices))
        all_p, all_st = [], []

        for pos, m in loop:
            kw = fixed_kw if exists(fixed_kw) else self._filter_kwargs(self._resolve_kwargs(index = m), differentiable = create_graph)
            step_p = [_work_tensor(p if not exists(m) else p[m], inplace, create_graph) for _, p, _ in items]
            step_g = [g if not exists(m) else g[pos] for _, _, g in items]
            if inplace:
                step_g = [g.detach() for g in step_g]
            step_st = [
                self._param_state(store, s, m, views, clone = not inplace, clone_shared = exists(indices), to_cpu = True)
                if self._is_init(store, s, m) else {}
                for s, _, _ in items
            ]
            all_st.append(self._run_inner(step_p, step_g, step_st, kw))
            all_p.append(step_p)

        flags, tidxs = store['global/_init'], [s.trainable_index for s, _, _ in items]
        for _, m in loop:
            if flags.ndim == 1:
                flags[tidxs] = True
            else:
                flags[slice(None) if not exists(m) else m, tidxs] = True

        next_views = None if inplace else {k: dict(v) for k, v in views.items()}
        for (spec, _, _), pstates in zip(items, zip(*all_st)):
            for k in {k for st in pstates for k, v in st.items() if is_tensor(v)}:
                vals = [st[k] for st in pstates]
                is_sh_key = exists(indices) and not spec.shared and self._is_key_shared(k)
                if inplace:
                    if is_sh_key:
                        self._write_state_(store, spec, k, reduce_mean(torch.stack(vals), vals[0].dtype), None, views)
                    else:
                        for m, v in zip([None] if not exists(indices) else indices, vals):
                            self._write_state_(store, spec, k, v, m, views)
                    continue
                stacked = torch.stack(vals)
                val = reduce_mean(stacked, vals[0].dtype) if is_sh_key else stacked
                if k in store.slots:
                    next_views[k][spec.name] = val
                elif (k, spec.name) in store.aux:
                    ak = f'aux/{k}/{spec.name}'
                    store.set(ak, val.to(store[ak].device), populated = not is_sh_key)

        if not inplace:
            for k, slot in store.slots.items():
                if slot.pop_names:
                    sk = f'slot/{k}/pop'
                    is_pop = store[sk].ndim == 2
                    store.set(sk, pack([next_views[k][n] for n in slot.pop_names], 'p *' if is_pop else '*')[0], populated = is_pop)
        return all_p

    def step(self, closure = None, *, grads: Mapping[str, Tensor] | Tensor | None = None, members = None):
        loss = None
        if exists(closure):
            with torch.enable_grad():
                loss = closure()

        pp, store = self._pp, self._store
        active_idx = self._member_indices(members)

        with torch.no_grad():
            if exists(self._try_batched_step(pp, store, pp._pop_leaf, grads, inplace = True, active_idx = active_idx)):
                return loss

            schema = pp._schema
            named_p = pp.named_parameters_dict()
            named_g = pp.unpack_flat(grads) if is_tensor(grads) else (grads if exists(grads) else pp.named_grads_dict())

            shared_specs = schema.shared_trainable if pp.is_population else schema.trainable
            groups: list[tuple[list[TensorSpec], list[int] | None]] = [(shared_specs, None)]
            if pp.is_population:
                groups.append((schema.pop_trainable, default(active_idx, list(range(pp.pop_size)))))

            for specs, idx in groups:
                items = [(s, named_p[s.name], _match_grad(s, named_g[s.name], active_idx, pp.pop_size)) for s in specs if exists(named_g.get(s.name))]
                if not items:
                    continue
                fixed_kw = None if exists(idx) else self._filter_kwargs(self._resolve_kwargs(index = active_idx, collapse = pp.is_population))
                self._step_group(store, items, idx, inplace = True, fixed_kw = fixed_kw)
        return loss

    def functional_step(self, loss: Tensor, params: PopulationParameters | None = None, *, create_graph: bool = True, retain_graph: bool | None = None) -> PopulationParameters:
        pp = default(params, self._pp)
        if not pp.is_population:
            raise ValueError('functional_step requires a populated PopulationParameters')

        pop_leaf = pp._pop_leaf
        flat_g = None
        if exists(pop_leaf):
            (flat_g,) = torch_grad(loss, (pop_leaf,), only_inputs = True, create_graph = create_graph, retain_graph = default(retain_graph, create_graph), allow_unused = True)
            if exists(flat_g) and not create_graph:
                flat_g = flat_g.detach()

        next_pp = pp._spawn_functional(pop_leaf, pp._shared_leaf)
        next_store = next_pp.opt_stores[self.name]
        next_store.set('global/_init', next_store['global/_init'].clone(), populated = True)

        with (torch.enable_grad() if create_graph else torch.no_grad()):
            next_pop = self._try_batched_step(next_pp, next_store, pop_leaf, flat_g, inplace = False, create_graph = create_graph)
            if not exists(next_pop) and exists(flat_g):
                items = [(s, slice_spec(s, pop_leaf), slice_spec(s, flat_g)) for s in pp._schema.pop_trainable]
                all_p = self._step_group(next_store, items, list(range(pp.pop_size)), inplace = False, create_graph = create_graph)
                packed = pack([torch.stack(ts) for ts in zip(*all_p)], 'p *')[0]
                next_pop = packed if create_graph else packed.requires_grad_(True)
            if exists(next_pop):
                next_pp._func_leaves = (next_pop, next_pp._func_leaves[1])
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
        cpu_keys = {pair for pair, meta in store.aux.items() if meta.cpu_scalar}

        store.clear(pred = lambda k: not k.startswith('hp/'))
        store.slots.clear()
        store.aux.clear()
        self._alloc_from_probe(store, old_state, cpu_keys)

        views, flags = store.all_views(), store['global/_init']
        for s in pp._schema.trainable:
            for k, v in old_state[s.name].items():
                buf = views[k][s.name] if k in store.slots else store[f'aux/{k}/{s.name}']
                buf.copy_(expand(v) if buf.ndim == v.ndim + 1 else v)
            flag = old_init[s.trainable_index]
            flags[:, s.trainable_index] = flag if s.shared else expand(flag)
        self._sync_optimizer_state()

    def _clone_for_subpopulation(self, target_pop: Population, indices: list[int]) -> PopulationOptimizer:
        is_pop = self.population.is_population
        kw = self._resolve_kwargs(index = indices if is_pop else None)
        dst_pp = target_pop.population_parameters

        if not is_pop:
            sub_store = self._store.clone_handle(dst_pp)
            dst_pp.opt_stores[self.name] = sub_store
            for k, t in self._store.items():
                sub_store.set(k, t.detach().clone(), populated = False)

        sub = PopulationOptimizer.__new__(PopulationOptimizer)
        sub.population = target_pop
        for attr in ('opt_cls', '_opt_args', 'name', '_share_all', '_accepted_kwargs', '_can_batch'):
            setattr(sub, attr, getattr(self, attr))
        sub._shared_keys = set(self._shared_keys)
        dst_pp._optimizers[self.name] = sub

        Optimizer.__init__(sub, sub._leaf_params(), kw)
        sub._sync_optimizer_state()
        if not is_pop:
            sub._on_populate(len(indices), init = 'clone', preserve_first = True)
        return sub

    @torch.no_grad()
    def _copy_from_other_opt_(self, dst_idx: list[int], src_opt: PopulationOptimizer, src_idx: list[int]):
        if [s.name for s in self._pp._schema.trainable] != [s.name for s in src_opt._pp._schema.trainable]:
            raise ValueError('source optimizer parameter names do not match target optimizer')

        for k, dst in self._store.items():
            if k.startswith('hp/') or k not in src_opt._store:
                continue
            src = src_opt._store[k]
            src_pop = src_opt._store.is_populated(k, src_opt._pp.pop_size)
            if self._store.is_populated(k, self._pp.pop_size):
                dst[dst_idx] = (src[src_idx] if src_pop else src).to(dst)
            else:
                dst.copy_((src[src_idx[-1]] if src_pop else src).to(dst))

        if self._pp.is_population:
            hp_src_idx = src_idx if src_opt._pp.is_population else None
            _copy_hp_(self._store, src_opt._resolve_kwargs(), dst_idx, hp_src_idx)
