from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, ClassVar, Literal

import torch
from einops import pack, rearrange, reduce, repeat, unpack
from torch import Tensor, is_tensor
from torch.autograd import grad as torch_grad
from torch.func import vmap
from torch.nn import Module
from torch.optim import Optimizer

from torch_population.parameters import PopulationParameters
from torch_population.store import (
    FunctionalOptState,
    OptStore,
    ParamSlot,
    TensorSpec,
    default,
    exists,
    is_int_index,
    matches_pattern,
    slice_pop,
    slice_spec,
    unpack_tensors,
)

if TYPE_CHECKING:
    from torch_population.torch_population import Population

# helpers

_SLOT_ALIASES = {
    'm': 'm', 'moment': 'm', 'moments': 'm', 'exp_avg': 'm',
    'v': 'v', 'variance': 'v', 'variances': 'v', 'exp_avg_sq': 'v',
    'step': 'step', 'steps': 'step', 't': 'step', 'time': 'step',
    'momentum': 'momentum', 'momentum_buffer': 'momentum', 'buf': 'momentum',
    'm1': 'm1', 'm2': 'm2',
}

def normalize_shared_slots(
    shared: bool | str | Iterable[str],
    valid: frozenset[str] = frozenset({'m', 'v', 'step'}),
) -> frozenset[str]:
    if isinstance(shared, bool):
        return valid if shared else frozenset()

    if isinstance(shared, str):
        key = shared.strip().lower()
        if key in ('all', 'true'):
            return valid
        if key in ('none', 'false', ''):
            return frozenset()
        mapped = _SLOT_ALIASES.get(key)
        if mapped in valid:
            return frozenset({mapped})
        raise ValueError(f'unknown shared slot {shared!r}; expected bool, all, none, or subset of {sorted(valid)}')

    slots: set[str] = set()
    for item in shared:
        if not isinstance(item, str):
            raise TypeError(f'shared slot names must be strings, got {type(item).__name__}')
        mapped = _SLOT_ALIASES.get(item.strip().lower())
        if mapped not in valid:
            raise ValueError(f'unknown shared slot {item!r}; expected subset of {sorted(valid)}')
        slots.add(mapped)
    return frozenset(slots)

def _coerce_hp(
    value: float | Tensor | Sequence[float],
    pop_size: int | None,
    name: str,
    *,
    device,
    dtype,
) -> float | Tensor:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)

    if is_tensor(value):
        t = value.detach().to(device = device, dtype = dtype)
        if t.ndim == 0 or (not exists(pop_size) and t.ndim == 1 and t.numel() == 1):
            return float(t.item())
        if not exists(pop_size):
            raise ValueError(f'per-individual {name} requires a populated Population')
        if t.ndim != 1 or t.shape[0] != pop_size:
            raise ValueError(f'per-individual {name} must have shape ({pop_size},), got {tuple(t.shape)}')
        return t.clone()

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if not exists(pop_size):
            if len(value) == 1:
                return float(value[0])
            raise ValueError(f'per-individual {name} requires a populated Population')
        if len(value) != pop_size:
            raise ValueError(f'per-individual {name} must have length {pop_size}, got {len(value)}')
        return torch.tensor([float(v) for v in value], device = device, dtype = dtype)

    raise TypeError(f'{name} must be a float, 1D Tensor, or sequence of floats, got {type(value).__name__}')

def _slice_hp(val: float | Tensor, idx: list[int] | int) -> float | Tensor:
    if not is_tensor(val):
        return val
    return float(val[idx].item()) if isinstance(idx, int) else val[idx]

def _bcast_pop(val: float | Tensor, ndim: int, *, device, dtype) -> float | Tensor:
    if not is_tensor(val):
        return val
    t = val.to(device = device, dtype = dtype)
    if t.ndim == 0:
        return t
    return t.view(t.shape[0], *([1] * (ndim - 1)))

def _advance_steps(steps: Tensor, ks, active_idx: list[int] | None, *, ndim: int, dtype) -> float | Tensor:
    k0 = ks[0] if isinstance(ks, list) else ks
    if steps.ndim == 2:
        row = slice(None) if not exists(active_idx) else active_idx
        steps[row, ks] += 1
        return _bcast_pop(steps[row, k0], ndim, device = steps.device, dtype = dtype)
    steps[ks] += 1
    return float(steps[k0].item())

def newton_schulz(x: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    a, b, c = 3.4445, -4.7750, 2.0315
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.mT

    x = x / x.norm().clamp(min = eps)
    for _ in range(steps):
        xx = x @ x.mT
        x = a * x + (b * xx + c * xx @ xx) @ x

    return x.mT if transposed else x

def orthogonalize(x: Tensor, *, is_pop: bool, steps: int = 5) -> Tensor:
    if x.ndim < 2:
        return x
    if is_pop:
        mat, ps = pack([x], 'p r *')
        orth = vmap(lambda m: newton_schulz(m, steps = steps))(mat)
        return unpack(orth, ps, 'p r *')[0]
    mat, ps = pack([x], 'r *')
    return unpack(newton_schulz(mat, steps = steps), ps, 'r *')[0]

# general colocated population optimizer engine

class PopulationOptimizer(Optimizer):
    """General optimizer engine colocated inside PopulationParameters.

    Subclasses only declare `_init_store_slots` and a pure `_update(p, g, slots, step, hps, ...)` rule.
    Noop mode, vectorized population stepping, shared/granular slots, member subsets, shared params,
    extract/writeback, lifecycle resizing, and differentiable `functional_step` all work automatically.
    """

    _valid_slots: ClassVar[frozenset[str]] = frozenset()
    _slot_to_torch: ClassVar[dict[str, str]] = {}
    _per_param: ClassVar[bool] = False

    def __init__(
        self,
        population: Population,
        *,
        shared: bool | str | Iterable[str] = False,
        name: str = 'opt',
        hp_specs: dict[str, float | Tensor | Sequence[float]],
        defaults: dict,
    ):
        from torch_population.torch_population import Population as PopCls
        if not isinstance(population, PopCls):
            raise TypeError(f'{type(self).__name__} requires a Population instance, got {type(population).__name__}')

        pp = population.population_parameters
        if not pp.trainable_specs:
            raise ValueError('Population has no trainable parameters')

        self._pop_ref = [population]
        self.name = name
        self._shared_slots = normalize_shared_slots(shared, self._valid_slots)
        self._param_names = list(pp.trainable_param_names)

        store = pp.register_opt_store(name)
        self._init_store_slots(store)

        first_t = pp.named_parameters_dict()[self._param_names[0]]
        kw = dict(device = first_t.device, dtype = first_t.dtype)

        coerced = {}
        for hp_name, raw in hp_specs.items():
            val = _coerce_hp(raw, population.pop_size, hp_name, **kw)
            coerced[hp_name] = val
            if is_tensor(val):
                store.set_hp(hp_name, val)

        self._apply_coerced_hps(defaults, coerced)
        population._register_optimizer(name, self)

        super().__init__(self._optimizer_leaf_params(), defaults)
        self._sync_optimizer_state()

    # subclass hooks

    def _init_store_slots(self, store: OptStore):
        raise NotImplementedError

    def _update(
        self,
        p: Tensor,
        g: Tensor,
        slots: dict[str, Tensor],
        step: float | Tensor | None,
        hps: dict[str, float | Tensor],
        *,
        spec: TensorSpec | None = None,
        is_pop: bool = False,
        stop_grad_denom: bool = False,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        raise NotImplementedError

    def _spawn_sub_optimizer(self, target_pop: Population, indices: list[int]) -> PopulationOptimizer:
        raise NotImplementedError

    def _apply_coerced_hps(self, defaults: dict, coerced: dict):
        defaults.update(coerced)

    def _hp_names(self) -> tuple[str, ...]:
        return ('lr', 'weight_decay')

    def _resolve_hp_dict(self) -> dict[str, float | Tensor]:
        grp, store = self.param_groups[0], self._store
        return {k: default(store.hp_buffer(k), grp[k]) for k in self._hp_names()}

    def _alloc_step_globals(self, store: OptStore):
        pp = self._pp
        first_t = pp.named_parameters_dict()[self._param_names[0]]
        num_pop = len(pp.pop_trainable_specs) if pp.is_population else len(pp.trainable_specs)
        is_pop_step = pp.is_population and 'step' not in self._shared_slots
        step_shape = (pp.pop_size, num_pop) if is_pop_step else (num_pop,)
        store.alloc_global('steps', torch.zeros(step_shape, device = first_t.device, dtype = torch.int64), populated = is_pop_step)

        if pp.is_population and pp.shared_trainable_specs:
            sh_steps = torch.zeros((len(pp.shared_trainable_specs),), device = first_t.device, dtype = torch.int64)
            store.alloc_global('shared_steps', sh_steps, populated = False)

    # core properties & optimizer sync

    def _optimizer_leaf_params(self) -> list[Tensor]:
        pp = self._pp
        if pp.is_population:
            t_flats, st_flats = pp.param_leaves()
            return t_flats + st_flats
        named = pp.named_parameters_dict()
        return [named[n] for n in self._param_names]

    @property
    def population(self) -> Population:
        return self._pop_ref[0]

    @property
    def _pp(self) -> PopulationParameters:
        return self.population.population_parameters

    @property
    def _store(self) -> OptStore:
        return self._pp.opt_store(self.name)

    @property
    def shared_slots(self) -> frozenset[str]:
        return self._shared_slots

    @property
    def is_shared(self) -> bool:
        return bool(self._valid_slots) and self._shared_slots == self._valid_slots

    def _sync_optimizer_state(self):
        pp, store = self._pp, self._store
        grp = self.param_groups[0]
        grp['params'] = self._optimizer_leaf_params()
        self._apply_coerced_hps(grp, self._resolve_hp_dict())

        self.state.clear()
        if pp.is_population:
            t_flats, st_flats = pp.param_leaves()
            for flats, is_sh, step_key in ((t_flats, False, 'steps'), (st_flats, True, 'shared_steps')):
                for b_idx, fp in enumerate(flats):
                    entry = {}
                    if store.has_global(step_key):
                        entry['step'] = store.global_tensor(step_key)
                    for sn in store.slot_names():
                        s_flats = store.slot_flats(sn, shared_params = is_sh)
                        if b_idx < len(s_flats):
                            entry[self._slot_to_torch.get(sn, sn)] = s_flats[b_idx]
                    self.state[fp] = entry
        else:
            named_p = pp.named_parameters_dict()
            named_opt = self.named_state(pp)
            for pname in self._param_names:
                if pname in named_opt:
                    self.state[named_p[pname]] = named_opt[pname]

    # generic state inspection, PyTorch state_dict & solo extract/writeback

    def named_state(
        self,
        pp: PopulationParameters,
        *,
        idx: int | None = None,
        clone: bool = False,
    ) -> dict[str, dict[str, Tensor]]:
        maybe = (lambda t: t.detach().clone()) if clone else (lambda t: t)
        out: dict[str, dict[str, Tensor]] = {s.name: {} for s in pp.trainable_specs}

        if pp.is_functional:
            fstate = pp._functional_opt_states[self.name]
            steps = fstate.globals.get('steps')
            sh_steps = fstate.globals.get('shared_steps')
            for sn, flats in fstate.slots.items():
                tk = self._slot_to_torch.get(sn, sn)
                sh_flats = fstate.shared_slots.get(sn, [])
                for s in pp.trainable_specs:
                    if s.shared and s.bucket < len(sh_flats):
                        out[s.name][tk] = maybe(slice_spec(s, sh_flats[s.bucket]))
                    elif not s.shared and s.bucket < len(flats):
                        out[s.name][tk] = maybe(slice_spec(s, flats[s.bucket], idx = idx, pop_size = pp.pop_size))
        else:
            store: OptStore = pp.opt_stores[self.name]
            steps = store.global_tensor('steps') if store.has_global('steps') else None
            sh_steps = store.global_tensor('shared_steps') if store.has_global('shared_steps') else None
            for sn in store.slot_names():
                tk = self._slot_to_torch.get(sn, sn)
                for pname, view in store.slot_views(sn, idx = idx).items():
                    if pname in out:
                        out[pname][tk] = maybe(view)

        if exists(steps):
            for s in pp.trainable_specs:
                k = s.trainable_index
                if s.shared and exists(sh_steps):
                    sv = sh_steps[k]
                elif steps.ndim == 2:
                    sv = steps[idx, k] if exists(idx) else steps[:, k]
                else:
                    sv = steps[k]
                out[s.name]['step'] = maybe(sv)

        return out

    def _param_group_dict(self, num_params: int, index: int | None = None) -> dict:
        grp = self.param_groups[0]
        out = {}
        for k, v in grp.items():
            if k == 'params':
                continue
            if isinstance(v, tuple):
                out[k] = tuple(_slice_hp(x, index) if exists(index) else x for x in v)
            else:
                out[k] = _slice_hp(v, index) if exists(index) else v
        out['params'] = list(range(num_params))
        return out

    def state_dict_for(self, pp: PopulationParameters, *, idx: int | None = None, clone: bool = False) -> dict:
        named_opt = self.named_state(pp, idx = idx, clone = clone)
        t_specs = pp.trainable_specs
        return {
            'state': {i: named_opt[s.name] for i, s in enumerate(t_specs)},
            'param_groups': [self._param_group_dict(len(t_specs), index = idx)],
        }

    def individual_state(
        self,
        pp: PopulationParameters,
        idx: int | None,
        *,
        clone: bool = False,
    ) -> tuple[dict[str, dict[str, Tensor]], dict]:
        named_opt = self.named_state(pp, idx = idx, clone = clone)
        t_specs = pp.trainable_specs
        opt_sd = {
            'state': {i: named_opt[s.name] for i, s in enumerate(t_specs)},
            'param_groups': [self._param_group_dict(len(t_specs), index = idx)],
        }
        return named_opt, opt_sd

    def load_individual_state_dict_(self, pp: PopulationParameters, idx: int, data: Mapping):
        store: OptStore = pp.opt_stores[self.name]
        named_src = data.get('named_optimizer_state')
        opt_sd = data.get('optimizer_state')
        indexed_src = opt_sd.get('state', {}) if exists(opt_sd) else {}

        steps = store.global_tensor('steps') if store.has_global('steps') else None
        sh_steps = store.global_tensor('shared_steps') if store.has_global('shared_steps') else None
        slot_views = {sn: store.slot_views(sn) for sn in store.slot_names()}
        slot_shared = {sn: store.slot_is_shared(sn) for sn in store.slot_names()}

        with torch.no_grad():
            for py_i, s in enumerate(pp.trainable_specs):
                k = s.trainable_index
                entry = named_src.get(s.name) if exists(named_src) else indexed_src.get(py_i)

                for sn, views in slot_views.items():
                    if s.name not in views:
                        continue
                    tk = self._slot_to_torch.get(sn, sn)
                    buf = views[s.name]
                    is_pop_buf = pp.is_population and not s.shared and not slot_shared[sn]
                    dst = buf[idx] if is_pop_buf else buf
                    if exists(entry) and tk in entry:
                        dst.copy_(entry[tk].detach().to(dst))
                    elif is_pop_buf:
                        dst.zero_()

                if exists(steps):
                    st_t = sh_steps if (s.shared and exists(sh_steps)) else steps
                    has_state = exists(entry) and bool(entry)
                    step_val = round(float(entry.get('step', 1))) if has_state else 0
                    if st_t.ndim == 2:
                        st_t[idx, k] = step_val
                    elif has_state or not pp.is_population:
                        st_t[k] = step_val

            if pp.is_population and exists(opt_sd) and opt_sd.get('param_groups'):
                self._load_group_hps_(store, idx, opt_sd['param_groups'][0])

    def _load_group_hps_(self, store: OptStore, idx: int, grp: dict):
        for hp_name in self._hp_names():
            buf = store.hp_buffer(hp_name)
            if exists(buf) and hp_name in grp and exists(grp[hp_name]):
                buf[idx] = float(grp[hp_name])

    def slot(self, name: str) -> ParamSlot:
        if name not in self._param_names:
            raise KeyError(f'no tracked trainable parameter named {name!r}')
        pp = self._pp
        entry = self.named_state(pp)[name]
        return ParamSlot(
            name = name,
            param = pp.named_parameters_dict()[name],
            grad = pp.named_grads_dict()[name],
            exp_avg = entry.get('exp_avg'),
            exp_avg_sq = entry.get('exp_avg_sq'),
            step = entry.get('step'),
            state = entry,
        )

    def _build_extracted_optimizer(self, ind: Module, grp: dict) -> Optimizer:
        raise NotImplementedError(f'extract() is not supported for {type(self).__name__}')

    def _format_extracted_state(self, p: Tensor, entry: dict[str, Tensor]) -> dict[str, Tensor]:
        return {
            k: (torch.tensor(float(v.item()), dtype = torch.float32) if k == 'step' else v.to(p))
            for k, v in entry.items()
        }

    def extract(self, index: int = 0) -> tuple[Module, Optimizer]:
        pop, pp = self.population, self._pp
        idx = pop._normalize_index(index)
        ind = pop[idx]
        ind_data = pp.individual(idx, opt = self.name, clone = True)
        ind_opt = self._build_extracted_optimizer(ind, ind_data['optimizer_state']['param_groups'][0])

        ind_named = dict(ind.named_parameters(remove_duplicate = True))
        named_opt = ind_data['named_optimizer_state']
        for pname in self._param_names:
            p = ind_named[pname]
            formatted = self._format_extracted_state(p, named_opt.get(pname, {}))
            if formatted:
                ind_opt.state[p] = formatted
        return ind, ind_opt

    def load_individual_state_(self, index: int, module: Module, optimizer: Optimizer):
        mod_named = dict(module.named_parameters(remove_duplicate = True))
        named_opt_state = {}
        for pname in self._param_names:
            if pname not in mod_named:
                raise ValueError(f'source module is missing parameter {pname!r}')
            st = optimizer.state.get(mod_named[pname])
            if exists(st) and st:
                named_opt_state[pname] = st

        payload: dict = {'named_optimizer_state': named_opt_state}
        if optimizer.param_groups:
            payload['optimizer_state'] = {'param_groups': optimizer.param_groups}
        self._pp.load_individual_dict_(index, payload, opt = self.name)

    # runtime share/unshare, reset & zero_grad

    def _convert_sharing(self, slots: bool | str | Iterable[str], *, to_shared: bool):
        target = normalize_shared_slots(slots, self._valid_slots)
        if not target:
            return self

        pp, store = self._pp, self._store
        if pp.is_population:
            p_size = pp.pop_size
            with torch.no_grad():
                for sk in store.slot_names():
                    if sk not in target:
                        continue
                    for b, t in enumerate(store.slot_flats(sk)):
                        if to_shared and t.ndim == 2:
                            store.set_slot_flat(sk, b, reduce(t, 'p d -> d', 'mean').clone())
                        elif not to_shared and t.ndim == 1:
                            store.set_slot_flat(sk, b, repeat(t, 'd -> p d', p = p_size).clone())

                if 'step' in target and store.has_global('steps'):
                    st = store.global_tensor('steps')
                    if to_shared and st.ndim == 2:
                        store.set_global('steps', reduce(st.float(), 'p k -> k', 'mean').round().to(torch.int64), populated = False)
                    elif not to_shared and st.ndim == 1:
                        store.set_global('steps', repeat(st, 'k -> p k', p = p_size).clone(), populated = True)

        self._shared_slots = (self._shared_slots | target) if to_shared else (self._shared_slots - target)
        self._sync_optimizer_state()
        return self

    def share(self, slots: bool | str | Iterable[str] = True):
        return self._convert_sharing(slots, to_shared = True)

    def unshare(self, slots: bool | str | Iterable[str] = True):
        return self._convert_sharing(slots, to_shared = False)

    def _member_indices(self, members) -> list[int] | None:
        if not exists(members):
            return None
        pop = self.population
        if not pop.is_population:
            pop._normalize_index(members)
            return None
        raw = [pop._normalize_index(members)] if is_int_index(members) else pop._normalize_indices(members)
        return list(dict.fromkeys(raw))

    def reset(self, members = None):
        self._store.reset_members_(self._member_indices(members))
        return self

    def zero_grad(self, set_to_none: bool = True, *, members = None):
        idx = self._member_indices(members)
        if not exists(idx):
            super().zero_grad(set_to_none = set_to_none)
            self._pp.clear_grad_tracking()
            return
        with torch.no_grad():
            for flat_p in self._pp.param_leaves()[0]:
                if exists(flat_p.grad):
                    flat_p.grad[idx] = 0.

    # unified step engine (handles noop, flat buckets, per-param shapes, shared slots, subsets, shared params)

    def _apply_update_inplace_(
        self,
        p: Tensor,
        g: Tensor,
        slot_bufs: dict[str, Tensor],
        step: float | Tensor | None,
        hps: dict[str, float | Tensor],
        *,
        spec: TensorSpec | None = None,
        is_pop: bool = False,
    ):
        expanded = {k: (b.unsqueeze(0) if b.ndim == p.ndim - 1 else b) for k, b in slot_bufs.items()}
        p_next, next_slots = self._update(p, g, expanded, step, hps, spec = spec, is_pop = is_pop)
        p.copy_(p_next)
        for k, b in slot_bufs.items():
            if k in next_slots:
                b.copy_(next_slots[k].mean(dim = 0) if b.ndim == p.ndim - 1 else next_slots[k])

    def _slice_hps(
        self,
        raw_hps: dict[str, float | Tensor],
        active_idx: list[int] | None,
        *,
        ndim: int,
        device,
        dtype,
        collapse: bool = False,
    ) -> dict[str, float | Tensor]:
        out = {}
        for k, v in raw_hps.items():
            if collapse:
                out[k] = float((v[active_idx] if exists(active_idx) else v).mean().item()) if is_tensor(v) else float(v)
            else:
                sliced = _slice_hp(v, active_idx) if exists(active_idx) else v
                out[k] = _bcast_pop(sliced, ndim, device = device, dtype = dtype)
        return out

    def step(
        self,
        closure = None,
        *,
        grads: Mapping[str, Tensor] | Tensor | None = None,
        lane: str | None = None,
        members = None,
    ):
        loss = None
        if exists(closure):
            with torch.enable_grad():
                loss = closure()
        if exists(lane) and exists(grads):
            raise ValueError('pass either `grads` or `lane`, not both')

        pp = self._pp
        if exists(lane):
            grads = pp.lane(lane)

        active_idx = self._member_indices(members)
        raw_hps = self._resolve_hp_dict()

        with torch.no_grad():
            if not pp.is_population:
                named_g = pp.unpack_flat(grads) if is_tensor(grads) else grads
                self._step_noop_(named_g, raw_hps)
            elif self._per_param:
                self._step_per_param_(grads, active_idx, raw_hps)
            else:
                self._step_flat_buckets_(grads, active_idx, raw_hps)
        return loss

    def _step_noop_(self, grads: Mapping[str, Tensor] | None, raw_hps: dict[str, float | Tensor]):
        pp, store = self._pp, self._store
        net_params = pp.named_parameters_dict()
        views = {sn: store.slot_views(sn) for sn in store.slot_names()}
        steps = store.global_tensor('steps') if store.has_global('steps') else None

        for s in pp.trainable_specs:
            p = net_params[s.name]
            g = grads.get(s.name) if exists(grads) else p.grad
            if not exists(g):
                continue

            step_val = None
            if exists(steps):
                steps[s.trainable_index] += 1
                step_val = float(steps[s.trainable_index].item())

            bufs = {sn: vm[s.name] for sn, vm in views.items() if s.name in vm}
            self._apply_update_inplace_(p, g, bufs, step_val, raw_hps, spec = s, is_pop = False)

    def _step_per_param_(self, grads, active_idx: list[int] | None, raw_hps: dict[str, float | Tensor]):
        pp, store = self._pp, self._store
        named_p = pp.named_parameters_dict()
        named_g = pp.unpack_flat(grads) if is_tensor(grads) else (grads if exists(grads) else pp.named_grads_dict())
        views = {sn: store.slot_views(sn) for sn in store.slot_names()}
        steps = store.global_tensor('steps') if store.has_global('steps') else None
        sh_steps = store.global_tensor('shared_steps') if store.has_global('shared_steps') else None

        for s in pp.trainable_specs:
            g_raw = named_g.get(s.name)
            if not exists(g_raw):
                continue

            p_full = named_p[s.name]
            k = s.trainable_index
            bufs_full = {sn: vm[s.name] for sn, vm in views.items() if s.name in vm}

            if s.shared:
                step_val = None
                if exists(sh_steps):
                    sh_steps[k] += 1
                    step_val = float(sh_steps[k].item())
                g_s = (g_raw[active_idx].sum(0) if exists(active_idx) else g_raw.sum(0)) if g_raw.ndim == len(s.shape) + 1 else g_raw
                hps = self._slice_hps(raw_hps, active_idx, ndim = p_full.ndim, device = p_full.device, dtype = p_full.dtype, collapse = True)
                self._apply_update_inplace_(p_full, g_s, bufs_full, step_val, hps, spec = s, is_pop = False)
                continue

            row = slice(None) if not exists(active_idx) else active_idx
            p_s = p_full[row]
            g_s = self._match_param_grad(s, g_raw, active_idx)
            step_val = _advance_steps(steps, k, active_idx, ndim = p_s.ndim, dtype = p_s.dtype) if exists(steps) else None
            hps = self._slice_hps(raw_hps, active_idx, ndim = p_s.ndim, device = p_s.device, dtype = p_s.dtype)

            bufs_sub = {sn: (b[row] if b.ndim == len(s.shape) + 1 else b) for sn, b in bufs_full.items()}
            self._apply_update_inplace_(p_s, g_s, bufs_sub, step_val, hps, spec = s, is_pop = True)

            if exists(active_idx):
                p_full[active_idx] = p_s
                for sn, b_sub in bufs_sub.items():
                    if bufs_full[sn].ndim == len(s.shape) + 1:
                        bufs_full[sn][active_idx] = b_sub

    def _match_param_grad(self, s: TensorSpec, gr: Tensor, active_idx: list[int] | None) -> Tensor:
        pp = self._pp
        if not exists(active_idx):
            return gr
        if gr.ndim == len(s.shape) + 1 and gr.shape[0] == pp.pop_size:
            return gr[active_idx]
        if gr.ndim == len(s.shape) + 1 and gr.shape[0] == len(active_idx):
            return gr
        if len(active_idx) == 1 and tuple(gr.shape) == s.shape:
            return gr.unsqueeze(0)
        raise ValueError(f'grad shape {tuple(gr.shape)} incompatible with {s.name!r} {s.shape}')

    def _step_flat_buckets_(self, grads, active_idx: list[int] | None, raw_hps: dict[str, float | Tensor]):
        pp, store = self._pp, self._store
        t_flats, st_flats = pp.param_leaves()
        slot_names = store.slot_names()
        steps = store.global_tensor('steps') if store.has_global('steps') else None

        for b_idx, flat_p in enumerate(t_flats):
            b_specs = [s for s in pp.pop_trainable_specs if s.bucket == b_idx]
            active_specs, flat_g, g_slices = self._extract_bucket_grads(b_specs, grads, flat_p, len(t_flats), active_idx)
            if not active_specs:
                continue

            hps = self._slice_hps(raw_hps, active_idx, ndim = 2, device = flat_p.device, dtype = flat_p.dtype)
            b_slots = {sn: store.slot_flats(sn)[b_idx] for sn in slot_names}

            ks = [s.trainable_index for s in active_specs]
            uniform = (
                not exists(steps)
                or (bool((steps[:, ks] == steps[:, ks[:1]]).all()) if steps.ndim == 2 else bool((steps[ks] == steps[ks[0]]).all()))
            )

            # fast fused full-bucket path when all specs active, full pop, and uniform steps
            if len(active_specs) == len(b_specs) and not exists(active_idx) and exists(flat_g) and uniform:
                step_val = _advance_steps(steps, ks, None, ndim = 2, dtype = flat_p.dtype) if exists(steps) else None
                self._apply_update_inplace_(flat_p, flat_g, b_slots, step_val, hps, is_pop = True)
                continue

            # per-spec flat slice path (handles unused heads, uneven per-param steps, or member subsets)
            row = slice(None) if not exists(active_idx) else active_idx
            for s in active_specs:
                lo, hi = s.offset, s.offset + s.numel
                step_val = _advance_steps(steps, s.trainable_index, active_idx, ndim = 2, dtype = flat_p.dtype) if exists(steps) else None
                p_s = flat_p[row, lo:hi]
                g_s = g_slices[s.name] if exists(g_slices) else flat_g[row, lo:hi]
                s_bufs = {sn: (bf[row, lo:hi] if bf.ndim == 2 else bf[lo:hi]) for sn, bf in b_slots.items()}

                self._apply_update_inplace_(p_s, g_s, s_bufs, step_val, hps, spec = s, is_pop = True)

                if exists(active_idx):
                    flat_p[active_idx, lo:hi] = p_s
                    for sn, bf in b_slots.items():
                        if bf.ndim == 2:
                            bf[active_idx, lo:hi] = s_bufs[sn]

        if not st_flats or is_tensor(grads):
            return

        sh_steps = store.global_tensor('shared_steps') if store.has_global('shared_steps') else None
        for sb_idx, s_flat_p in enumerate(st_flats):
            sb_specs = [s for s in pp.shared_trainable_specs if s.bucket == sb_idx]
            active_specs, s_flat_g, sg_map = self._extract_shared_grads(sb_specs, grads, s_flat_p, active_idx)
            if not active_specs:
                continue

            hps = self._slice_hps(raw_hps, active_idx, ndim = 1, device = s_flat_p.device, dtype = s_flat_p.dtype, collapse = True)
            sh_slots = {sn: store.slot_flats(sn, shared_params = True)[sb_idx] for sn in slot_names}

            for s in active_specs:
                k, lo, hi = s.trainable_index, s.offset, s.offset + s.numel
                step_val = None
                if exists(sh_steps):
                    sh_steps[k] += 1
                    step_val = float(sh_steps[k].item())
                g_s = sg_map[s.name] if exists(sg_map) else s_flat_g[lo:hi]
                s_bufs = {sn: bf[lo:hi] for sn, bf in sh_slots.items()}
                self._apply_update_inplace_(s_flat_p[lo:hi], g_s, s_bufs, step_val, hps, spec = s, is_pop = False)

    def _extract_bucket_grads(self, b_specs: list[TensorSpec], grads, flat_p: Tensor, num_buckets: int, active_idx: list[int] | None):
        pp = self._pp
        if is_tensor(grads):
            if num_buckets != 1:
                raise ValueError('passing a flat grad tensor requires a single trainable dtype bucket')
            return b_specs, grads, None

        if exists(grads):
            active = [s for s in b_specs if exists(grads.get(s.name))]
            n = len(active_idx) if exists(active_idx) else pp.pop_size
            g_slices = {s.name: self._match_param_grad(s, grads[s.name], active_idx).reshape(n, s.numel) for s in active}
            return active, None, g_slices

        if not exists(flat_p.grad):
            return [], None, None
        active = [s for s in b_specs if s.name in pp._active_grad_params] if pp._tracking_active_grads else b_specs
        return active, flat_p.grad, None

    def _extract_shared_grads(self, sb_specs: list[TensorSpec], grads, s_flat_p: Tensor, active_idx: list[int] | None):
        pp = self._pp
        if exists(grads):
            active = [s for s in sb_specs if exists(grads.get(s.name))]
            sg_map = {}
            for s in active:
                gr = grads[s.name]
                if gr.ndim == len(s.shape) + 1:
                    gr = gr[active_idx].sum(0) if exists(active_idx) else gr.sum(0)
                sg_map[s.name] = gr.reshape(s.numel)
            return active, None, sg_map

        if not exists(s_flat_p.grad):
            return [], None, None
        active = [s for s in sb_specs if s.name in pp._active_grad_params] if pp._tracking_active_grads else sb_specs
        return active, s_flat_p.grad, None

    # universal differentiable functional_step (works for Adam, AdamW, SGD, Muon, and custom optimizers!)

    def functional_step(
        self,
        loss: Tensor,
        params: PopulationParameters | None = None,
        *,
        create_graph: bool = True,
        retain_graph: bool | None = None,
        stop_grad_denom: bool = True,
    ) -> PopulationParameters:
        pp = default(params, self._pp)
        if not pp.is_population:
            raise ValueError('functional_step requires a populated PopulationParameters')

        t_flats, _ = pp.param_leaves()
        raw_hps = self._resolve_hp_dict()

        if pp.is_functional:
            fstate = pp._functional_opt_states[self.name]
            cur_slots = fstate.slots
            cur_shared = fstate.shared_slots
            steps = fstate.globals.get('steps')
            sh_steps = fstate.globals.get('shared_steps')
            hp_bufs = dict(fstate.hp_buffers)
        else:
            store = pp.opt_store(self.name)
            cur_slots = {sn: store.slot_flats(sn) for sn in store.slot_names()}
            cur_shared = {sn: store.slot_flats(sn, shared_params = True) for sn in store.slot_names()}
            steps = store.global_tensor('steps') if store.has_global('steps') else None
            sh_steps = store.global_tensor('shared_steps') if store.has_global('shared_steps') else None
            hp_bufs = store.hp_buffers_dict()

        grad_flats = torch_grad(
            loss, tuple(t_flats),
            only_inputs = True,
            create_graph = create_graph,
            retain_graph = default(retain_graph, create_graph),
            allow_unused = True,
        )
        if not create_graph:
            grad_flats = tuple(g.detach() if exists(g) else None for g in grad_flats)

        next_t: list[Tensor] = []
        next_slots: dict[str, list[Tensor]] = {sn: [] for sn in cur_slots}
        next_steps = steps.clone() if exists(steps) else None

        if self._per_param:
            # shape-aware functional path (e.g. Muon)
            slot_views = {}
            store_ref = self._store
            for sn, flats in cur_slots.items():
                slot_views[sn] = {}
                for b, flat in zip(store_ref._slots[sn]['pop'], flats):
                    slot_views[sn].update(zip(b.names, unpack_tensors(flat, b.shapes)))

            next_slot_views: dict[str, dict[str, Tensor]] = {sn: {} for sn in cur_slots}
            for b_idx, (flat_p, g_flat) in enumerate(zip(t_flats, grad_flats)):
                b_specs = [s for s in pp.pop_trainable_specs if s.bucket == b_idx]
                if not exists(g_flat) or not b_specs:
                    next_t.append(flat_p)
                    continue

                p_views = unpack_tensors(flat_p, [s.shape for s in b_specs])
                g_views = unpack_tensors(g_flat, [s.shape for s in b_specs])
                updated_ps = []
                for s, p_v, g_v in zip(b_specs, p_views, g_views):
                    step_val = _advance_steps(next_steps, s.trainable_index, None, ndim = p_v.ndim, dtype = p_v.dtype) if exists(next_steps) else None
                    hps = self._slice_hps(raw_hps, None, ndim = p_v.ndim, device = p_v.device, dtype = p_v.dtype)
                    s_in, is_sh = {}, {}
                    for sn, vm in slot_views.items():
                        if s.name in vm:
                            buf = vm[s.name]
                            is_sh[sn] = buf.ndim == p_v.ndim - 1
                            s_in[sn] = buf.unsqueeze(0) if is_sh[sn] else buf
                    p_n, ns = self._update(p_v, g_v, s_in, step_val, hps, spec = s, is_pop = True, stop_grad_denom = stop_grad_denom)
                    updated_ps.append(p_n)
                    for sn, val in ns.items():
                        next_slot_views[sn][s.name] = val.mean(dim = 0) if is_sh[sn] else val

                next_t.append(pack(updated_ps, 'p *')[0])

            for sn in cur_slots:
                for b, old_flat in zip(store_ref._slots[sn]['pop'], cur_slots[sn]):
                    vs = [next_slot_views[sn].get(n, slot_views[sn][n]) for n in b.names]
                    pat = '*' if old_flat.ndim == 1 else 'p *'
                    next_slots[sn].append(pack(vs, pat)[0])
        else:
            # fast fused flat-bucket functional path (Adam, AdamW, SGD, Lion, ...)
            for b_idx, (flat_p, g_flat) in enumerate(zip(t_flats, grad_flats)):
                b_specs = [s for s in pp.pop_trainable_specs if s.bucket == b_idx]
                if not exists(g_flat) or not b_specs:
                    next_t.append(flat_p)
                    for sn in cur_slots:
                        next_slots[sn].append(cur_slots[sn][b_idx])
                    continue

                dt = flat_p.dtype
                hps = self._slice_hps(raw_hps, None, ndim = 2, device = flat_p.device, dtype = dt)
                ks = [s.trainable_index for s in b_specs]

                step_val = None
                if exists(next_steps):
                    if next_steps.ndim == 2:
                        next_steps[:, ks] = next_steps[:, ks] + 1
                        if bool((next_steps[:, ks] == next_steps[:, ks[:1]]).all()):
                            step_val = rearrange(next_steps[:, ks[0]].to(dt), 'p -> p 1')
                        else:
                            cols = [repeat(next_steps[:, s.trainable_index].to(dt), 'p -> p n', n = s.numel) for s in b_specs]
                            step_val = torch.cat(cols, dim = 1)
                    else:
                        next_steps[ks] = next_steps[ks] + 1
                        if bool((next_steps[ks] == next_steps[ks[0]]).all()):
                            step_val = next_steps[ks[0]].to(dt)
                        else:
                            cols = [repeat(next_steps[s.trainable_index].to(dt), '-> p n', p = pp.pop_size, n = s.numel) for s in b_specs]
                            step_val = torch.cat(cols, dim = 1)

                s_in = {}
                for sn in cur_slots:
                    buf = cur_slots[sn][b_idx]
                    s_in[sn] = buf.unsqueeze(0) if buf.ndim == 1 else buf
                p_n, ns = self._update(flat_p, g_flat, s_in, step_val, hps, is_pop = True, stop_grad_denom = stop_grad_denom)
                next_t.append(p_n)
                for sn in cur_slots:
                    was_shared = cur_slots[sn][b_idx].ndim == 1
                    next_slots[sn].append(ns[sn].mean(dim = 0) if was_shared else ns[sn])

        new_globals = {}
        if exists(next_steps):
            new_globals['steps'] = next_steps
        if exists(sh_steps):
            new_globals['shared_steps'] = sh_steps.clone()

        new_opts = dict(pp._functional_opt_states) if pp.is_functional else {}
        new_opts[self.name] = FunctionalOptState(
            slots = next_slots,
            shared_slots = {sn: [t.clone() for t in fs] for sn, fs in cur_shared.items()},
            globals = new_globals,
            hp_buffers = hp_bufs,
        )
        return pp._spawn_functional(next_t, new_opts)

    # generic lifecycle hooks (populate, slice, copy)

    def _on_populate(self, pop_size: int, *, init: Literal['reinit', 'clone'] | object, preserve_first: bool):
        pp, store = self._pp, self._store

        def _expand(t: Tensor) -> Tensor:
            if init == 'clone':
                return repeat(t, '... -> p ...', p = pop_size).clone()
            out = torch.zeros((pop_size, *t.shape), device = t.device, dtype = t.dtype)
            if preserve_first:
                out[0].copy_(t)
            return out

        with torch.no_grad():
            old_slots = {sn: {k: v.clone() for k, v in store.slot_views(sn).items()} for sn in store.slot_names()}
            old_steps = dict(zip(self._param_names, store.global_tensor('steps').clone())) if store.has_global('steps') else None

            store.reset_slots_and_globals_()
            self._init_store_slots(store)

            for sn, old_views in old_slots.items():
                if not store.has_slot(sn):
                    continue
                new_views = store.slot_views(sn)
                for s in pp.trainable_specs:
                    if s.name in new_views and s.name in old_views:
                        v = old_views[s.name]
                        new_views[s.name].copy_(_expand(v) if (not s.shared and sn not in self._shared_slots) else v)

            if exists(old_steps) and store.has_global('steps'):
                steps = store.global_tensor('steps')
                sh_steps = store.global_tensor('shared_steps') if store.has_global('shared_steps') else None
                for s in pp.trainable_specs:
                    k, sv = s.trainable_index, old_steps[s.name]
                    if s.shared and exists(sh_steps):
                        sh_steps[k] = sv
                    elif steps.ndim == 2:
                        steps[:, k].copy_(_expand(sv))
                    else:
                        steps[k] = sv

        self._sync_optimizer_state()

    def _clone_for_subpopulation(self, target_pop: Population, indices: list[int]) -> PopulationOptimizer:
        sub_opt = self._spawn_sub_optimizer(target_pop, indices)
        src_s, dst_s, is_pop = self._store, sub_opt._store, self.population.is_population

        def _xfer(src_t: Tensor, dst_t: Tensor):
            if not is_pop:
                dst_t.copy_(repeat(src_t, '... -> p ...', p = dst_t.shape[0]) if dst_t.ndim == 2 else src_t)
            else:
                dst_t.copy_(slice_pop(src_t, indices) if src_t.ndim == 2 else src_t)

        with torch.no_grad():
            for sn in src_s.slot_names():
                for s_t, d_t in zip(src_s.slot_flats(sn), dst_s.slot_flats(sn)):
                    _xfer(s_t, d_t)
                for s_t, d_t in zip(src_s.slot_flats(sn, shared_params = True), dst_s.slot_flats(sn, shared_params = True)):
                    d_t.copy_(s_t)
            for gname, src_g in src_s.globals_dict().items():
                if dst_s.has_global(gname):
                    _xfer(src_g, dst_s.global_tensor(gname))
        return sub_opt

    def _copy_from_other_opt_(self, dst_indices: list[int], src_opt: PopulationOptimizer, src_indices: list[int]):
        if self._param_names != src_opt._param_names:
            raise ValueError('source optimizer parameter names do not match target optimizer')

        dst_s, src_s, pop, src_pop = self._store, src_opt._store, self.population, src_opt.population

        def _copy_pair(dst_t: Tensor, src_t: Tensor):
            if not pop.is_population:
                val = src_t[src_indices[0]] if (src_pop.is_population and src_t.ndim == 2) else src_t
                dst_t.copy_(val.to(dst_t))
                return
            for di, si in zip(dst_indices, src_indices):
                val = src_t[si] if (src_pop.is_population and src_t.ndim == 2) else src_t
                target = dst_t[di] if dst_t.ndim == 2 else dst_t
                target.copy_(val.to(target))

        with torch.no_grad():
            for sn in dst_s.slot_names():
                if src_s.has_slot(sn):
                    for d, s in zip(dst_s.slot_flats(sn), src_s.slot_flats(sn)):
                        _copy_pair(d, s)
            for gname, dst_g in dst_s.globals_dict().items():
                if src_s.has_global(gname):
                    _copy_pair(dst_g, src_s.global_tensor(gname))
            if pop.is_population:
                src_hps = src_opt._resolve_hp_dict()
                for hp_name in self._hp_names():
                    dst_hp = dst_s.hp_buffer(hp_name)
                    if not exists(dst_hp):
                        continue
                    src_val = src_hps[hp_name]
                    for di, si in zip(dst_indices, src_indices):
                        dst_hp[di] = (src_val[si] if src_pop.is_population else src_val) if is_tensor(src_val) else float(src_val)

# Adam & AdamW

class Adam(PopulationOptimizer):
    """Adam / AdamW with flat state colocated inside PopulationParameters."""

    _valid_slots: ClassVar[frozenset[str]] = frozenset({'m', 'v', 'step'})
    _slot_to_torch: ClassVar[dict[str, str]] = {'m': 'exp_avg', 'v': 'exp_avg_sq'}

    def __init__(
        self,
        population: Population,
        lr: float | Tensor | Sequence[float] = 1e-3,
        betas: tuple[float | Tensor | Sequence[float], float | Tensor | Sequence[float]] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float | Tensor | Sequence[float] = 0.,
        *,
        decoupled_weight_decay: bool = False,
        shared: bool | str | Iterable[str] = False,
        name: str = 'adam',
    ):
        self.eps = float(eps)
        self.decoupled_weight_decay = decoupled_weight_decay
        hp_specs = {'lr': lr, 'beta1': betas[0], 'beta2': betas[1], 'weight_decay': weight_decay}
        defaults = dict(lr = lr, betas = betas, eps = self.eps, weight_decay = weight_decay)
        super().__init__(population, shared = shared, name = name, hp_specs = hp_specs, defaults = defaults)

    def _init_store_slots(self, store: OptStore):
        store.alloc_slot('m', shared = 'm' in self._shared_slots)
        store.alloc_slot('v', shared = 'v' in self._shared_slots)
        self._alloc_step_globals(store)

    def _hp_names(self) -> tuple[str, ...]:
        return ('lr', 'beta1', 'beta2', 'weight_decay')

    def _apply_coerced_hps(self, defaults: dict, coerced: dict):
        defaults['lr'] = coerced['lr']
        defaults['betas'] = (coerced['beta1'], coerced['beta2'])
        defaults['weight_decay'] = coerced['weight_decay']

    def _resolve_hp_dict(self) -> dict[str, float | Tensor]:
        grp, store = self.param_groups[0], self._store
        return {
            'lr': default(store.hp_buffer('lr'), grp['lr']),
            'beta1': default(store.hp_buffer('beta1'), grp['betas'][0]),
            'beta2': default(store.hp_buffer('beta2'), grp['betas'][1]),
            'weight_decay': default(store.hp_buffer('weight_decay'), grp['weight_decay']),
        }

    def _param_group_dict(self, num_params: int, index: int | None = None) -> dict:
        hps = self._resolve_hp_dict()
        val = lambda x: _slice_hp(x, index) if exists(index) else x
        grp = {
            'lr': val(hps['lr']),
            'betas': (val(hps['beta1']), val(hps['beta2'])),
            'eps': self.eps,
            'weight_decay': val(hps['weight_decay']),
            'amsgrad': False, 'maximize': False, 'foreach': None,
            'capturable': False, 'differentiable': False, 'fused': None,
            'params': list(range(num_params)),
        }
        if not self.decoupled_weight_decay:
            grp['decoupled_weight_decay'] = False
        return grp

    def _load_group_hps_(self, store: OptStore, idx: int, grp: dict):
        vals = {'lr': grp.get('lr'), 'weight_decay': grp.get('weight_decay')}
        if 'betas' in grp:
            vals['beta1'], vals['beta2'] = grp['betas']
        for k, v in vals.items():
            buf = store.hp_buffer(k)
            if exists(buf) and exists(v):
                buf[idx] = float(v)

    def _build_extracted_optimizer(self, ind: Module, grp: dict) -> Optimizer:
        cls = torch.optim.AdamW if self.decoupled_weight_decay else torch.optim.Adam
        return cls(
            [p for p in ind.parameters() if p.requires_grad],
            lr = float(grp['lr']),
            betas = (float(grp['betas'][0]), float(grp['betas'][1])),
            eps = float(grp['eps']),
            weight_decay = float(grp['weight_decay']),
        )

    @property
    def exp_avg(self) -> dict[str, Tensor]:
        return {k: v['exp_avg'] for k, v in self.named_state(self._pp).items()}

    @property
    def exp_avg_sq(self) -> dict[str, Tensor]:
        return {k: v['exp_avg_sq'] for k, v in self.named_state(self._pp).items()}

    @property
    def steps(self) -> dict[str, Tensor]:
        return {k: v['step'] for k, v in self.named_state(self._pp).items()}

    moments = exp_avg
    variances = exp_avg_sq

    def _update(self, p, g, slots, step, hps, *, spec = None, is_pop = False, stop_grad_denom = False):
        lr, b1, b2, wd = hps['lr'], hps['beta1'], hps['beta2'], hps['weight_decay']
        if is_tensor(wd) or wd:
            if self.decoupled_weight_decay:
                p = p * (1. - lr * wd)
            else:
                g = g + wd * p

        m_next = slots['m'] * b1 + g * (1. - b1)
        gv = g.detach() if stop_grad_denom else g
        v_next = slots['v'] * b2 + (gv * gv) * (1. - b2)

        bc1 = 1. - b1 ** step
        bc2 = 1. - b2 ** step
        v_hat = v_next / bc2
        denom = (v_hat.detach() if stop_grad_denom else v_hat).sqrt() + self.eps
        return p - (lr / bc1) * (m_next / denom), {'m': m_next, 'v': v_next}

    def _spawn_sub_optimizer(self, target_pop: Population, indices: list[int]) -> Adam:
        hps = self._resolve_hp_dict()
        sub = lambda x: _slice_hp(x, indices).clone() if (self.population.is_population and is_tensor(x)) else x
        return type(self)(
            target_pop,
            lr = sub(hps['lr']),
            betas = (sub(hps['beta1']), sub(hps['beta2'])),
            eps = self.eps,
            weight_decay = sub(hps['weight_decay']),
            decoupled_weight_decay = self.decoupled_weight_decay,
            shared = self._shared_slots,
            name = self.name,
        )


class AdamW(Adam):
    """Decoupled weight decay AdamW colocated inside PopulationParameters."""

    def __init__(
        self,
        population: Population,
        lr: float | Tensor | Sequence[float] = 1e-3,
        betas: tuple[float | Tensor | Sequence[float], float | Tensor | Sequence[float]] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float | Tensor | Sequence[float] = 1e-2,
        *,
        decoupled_weight_decay: bool = True,
        shared: bool | str | Iterable[str] = False,
        name: str = 'adamw',
    ):
        super().__init__(
            population, lr = lr, betas = betas, eps = eps, weight_decay = weight_decay,
            decoupled_weight_decay = decoupled_weight_decay, shared = shared, name = name,
        )

# SGD with momentum & Nesterov

class SGD(PopulationOptimizer):
    """Colocated population SGD with momentum, dampening, Nesterov, and shared buffers."""

    _valid_slots: ClassVar[frozenset[str]] = frozenset({'momentum', 'step'})
    _slot_to_torch: ClassVar[dict[str, str]] = {'momentum': 'momentum_buffer'}

    def __init__(
        self,
        population: Population,
        lr: float | Tensor | Sequence[float] = 1e-2,
        momentum: float = 0.,
        dampening: float = 0.,
        weight_decay: float | Tensor | Sequence[float] = 0.,
        nesterov: bool = False,
        *,
        shared: bool | str | Iterable[str] = False,
        name: str = 'sgd',
    ):
        if nesterov and (momentum <= 0. or dampening != 0.):
            raise ValueError('Nesterov momentum requires momentum > 0 and dampening == 0')

        self.momentum = float(momentum)
        self.dampening = float(dampening)
        self.nesterov = bool(nesterov)
        defaults = dict(lr = lr, momentum = self.momentum, dampening = self.dampening, weight_decay = weight_decay, nesterov = self.nesterov)
        super().__init__(population, shared = shared, name = name, hp_specs = {'lr': lr, 'weight_decay': weight_decay}, defaults = defaults)

    def _init_store_slots(self, store: OptStore):
        if self.momentum:
            store.alloc_slot('momentum', shared = 'momentum' in self._shared_slots)
            self._alloc_step_globals(store)

    def _build_extracted_optimizer(self, ind: Module, grp: dict) -> Optimizer:
        return torch.optim.SGD(
            [p for p in ind.parameters() if p.requires_grad],
            lr = float(grp['lr']),
            momentum = self.momentum,
            dampening = self.dampening,
            weight_decay = float(grp['weight_decay']),
            nesterov = self.nesterov,
        )

    def _format_extracted_state(self, p: Tensor, entry: dict[str, Tensor]) -> dict[str, Tensor]:
        if not self.momentum or 'momentum_buffer' not in entry or ('step' in entry and int(entry['step'].item()) == 0):
            return {}
        return {'momentum_buffer': entry['momentum_buffer'].to(p)}

    def _update(self, p, g, slots, step, hps, *, spec = None, is_pop = False, stop_grad_denom = False):
        lr, wd = hps['lr'], hps['weight_decay']
        if is_tensor(wd) or wd:
            g = g + wd * p

        if not self.momentum:
            return p - lr * g, {}

        updated = slots['momentum'] * self.momentum + g * (1. - self.dampening)
        is_first = (step == 1)
        buf_next = torch.where(is_first, g, updated) if is_tensor(is_first) else (g if is_first else updated)
        u = g + self.momentum * buf_next if self.nesterov else buf_next
        return p - lr * u, {'momentum': buf_next}

    def _spawn_sub_optimizer(self, target_pop: Population, indices: list[int]) -> SGD:
        hps = self._resolve_hp_dict()
        sub = lambda x: _slice_hp(x, indices).clone() if (self.population.is_population and is_tensor(x)) else x
        return SGD(
            target_pop, lr = sub(hps['lr']), momentum = self.momentum, dampening = self.dampening,
            weight_decay = sub(hps['weight_decay']), nesterov = self.nesterov, shared = self._shared_slots, name = self.name,
        )

# Muon (orthogonalized Newton-Schulz on 2D+ matrices, AdamW on 1D/vector params)

class Muon(PopulationOptimizer):
    """Colocated population Muon (Newton-Schulz on 2D+ matrices, AdamW on 1D/vector params)."""

    _valid_slots: ClassVar[frozenset[str]] = frozenset({'m', 'm1', 'm2', 'step'})
    _slot_to_torch: ClassVar[dict[str, str]] = {'m': 'momentum_buffer', 'm1': 'exp_avg', 'm2': 'exp_avg_sq'}
    _per_param: ClassVar[bool] = True

    def __init__(
        self,
        population: Population,
        lr: float | Tensor | Sequence[float] = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        weight_decay: float | Tensor | Sequence[float] = 0.,
        *,
        adamw_lr: float | Tensor | Sequence[float] | None = None,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        no_decay = (),
        no_muon = (),
        ns_steps: int = 5,
        shared: bool | str | Iterable[str] = False,
        name: str = 'muon',
    ):
        self.momentum = float(momentum)
        self.nesterov = bool(nesterov)
        self.beta1, self.beta2 = float(betas[0]), float(betas[1])
        self.eps = float(eps)
        self.no_decay = no_decay
        self.no_muon = no_muon
        self.ns_steps = int(ns_steps)

        eff_adamw_lr = default(adamw_lr, lr)
        hp_specs = {'lr': lr, 'adamw_lr': eff_adamw_lr, 'weight_decay': weight_decay}
        defaults = dict(lr = lr, adamw_lr = eff_adamw_lr, weight_decay = weight_decay)
        super().__init__(population, shared = shared, name = name, hp_specs = hp_specs, defaults = defaults)

    def _is_matrix(self, spec: TensorSpec) -> bool:
        return len(spec.shape) >= 2 and not matches_pattern(spec.name, self.no_muon)

    def _hp_names(self) -> tuple[str, ...]:
        return ('lr', 'adamw_lr', 'weight_decay')

    def _init_store_slots(self, store: OptStore):
        pp = self._pp
        mat = [s.name for s in pp.trainable_specs if self._is_matrix(s)]
        vec = [s.name for s in pp.trainable_specs if not self._is_matrix(s)]
        if mat:
            store.alloc_slot('m', mat, shared = 'm' in self._shared_slots)
        if vec:
            store.alloc_slot('m1', vec, shared = 'm1' in self._shared_slots)
            store.alloc_slot('m2', vec, shared = 'm2' in self._shared_slots)
        self._alloc_step_globals(store)

    @staticmethod
    def _muon_scale(shape: tuple[int, ...]) -> float:
        rows = shape[0]
        cols = 1
        for d in shape[1:]:
            cols *= d
        return max(1., rows / cols) ** 0.5

    def _update(self, p, g, slots, step, hps, *, spec = None, is_pop = False, stop_grad_denom = False):
        lr, adamw_lr, wd = hps['lr'], hps['adamw_lr'], hps['weight_decay']
        if (is_tensor(wd) or wd) and not matches_pattern(spec.name, self.no_decay):
            p = p * (1. - lr * wd)

        if self._is_matrix(spec):
            m_next = slots['m'] * self.momentum + g
            u = g + self.momentum * m_next if self.nesterov else m_next
            orth = orthogonalize(u, is_pop = is_pop, steps = self.ns_steps) * self._muon_scale(spec.shape)
            return p - lr * orth, {'m': m_next}

        m1_next = slots['m1'] * self.beta1 + g * (1. - self.beta1)
        gv = g.detach() if stop_grad_denom else g
        m2_next = slots['m2'] * self.beta2 + (gv * gv) * (1. - self.beta2)
        bc1 = 1. - self.beta1 ** step
        bc2 = 1. - self.beta2 ** step
        v_hat = m2_next / bc2
        denom = (v_hat.detach() if stop_grad_denom else v_hat).sqrt() + self.eps
        return p - (adamw_lr / bc1) * (m1_next / denom), {'m1': m1_next, 'm2': m2_next}

    def _spawn_sub_optimizer(self, target_pop: Population, indices: list[int]) -> Muon:
        hps = self._resolve_hp_dict()
        sub = lambda x: _slice_hp(x, indices).clone() if (self.population.is_population and is_tensor(x)) else x
        return Muon(
            target_pop, lr = sub(hps['lr']), momentum = self.momentum, nesterov = self.nesterov,
            weight_decay = sub(hps['weight_decay']), adamw_lr = sub(hps['adamw_lr']),
            betas = (self.beta1, self.beta2), eps = self.eps, no_decay = self.no_decay,
            no_muon = self.no_muon, ns_steps = self.ns_steps, shared = self._shared_slots, name = self.name,
        )
