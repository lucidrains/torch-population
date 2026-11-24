from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
from typing import Literal

import torch
from torch import Tensor, is_tensor
from torch.func import functional_call, vmap
from torch.nn import Module
from torch.nn.utils.stateless import _reparametrize_module
from torch.optim import Optimizer
from torch.utils._pytree import tree_flatten, tree_unflatten

from torch_population.optimizer import PopulationOptimizer
from torch_population.parameters import PopulationParameters
from torch_population.store import (
    ParamSlot,
    clone_template,
    default,
    exists,
    init_module,
    is_int_index,
    is_sequence,
    reduce_pop,
)

# helpers

def check_pop_size(pop_size: int):
    if isinstance(pop_size, bool) or not isinstance(pop_size, int) or pop_size < 1:
        raise ValueError(f'pop_size must be a positive integer, got {pop_size!r}')

@torch.no_grad()
def add_param_noise_(module: Module, noise_std: float):
    for param in module.parameters():
        if param.is_floating_point():
            param.add_(torch.randn_like(param) * noise_std)

def invoke_module(module: Module, fn: str | Callable | None, args: tuple, kwargs: dict):
    if not exists(fn):
        return module(*args, **kwargs)
    if isinstance(fn, str):
        return getattr(module, fn)(*args, **kwargs)
    return fn(module, *args, **kwargs)

def stateless_call(
    module: Module,
    params: dict[str, Tensor],
    buffers: dict[str, Tensor],
    args: tuple,
    kwargs: dict,
    fn: str | Callable | None = None,
):
    if not exists(fn):
        return functional_call(module, (params, buffers), args, kwargs)
    with _reparametrize_module(module, {**params, **buffers}, tie_weights = True):
        return invoke_module(module, fn, args, kwargs)

def vmap_forward(
    template: Module,
    params: dict[str, Tensor],
    buffers: dict[str, Tensor],
    pop_size: int,
    args: tuple,
    kwargs: dict,
    *,
    fn: str | Callable | None = None,
    per_individual: bool = False,
    randomness: str = 'different',
):
    has_mapped = False
    input_dims = None

    if per_individual:
        leaves, spec = tree_flatten((args, kwargs))
        for leaf in leaves:
            if is_tensor(leaf) and (leaf.ndim == 0 or leaf.shape[0] != pop_size):
                raise ValueError(f'per_individual=True expects leading dim {pop_size}, got {tuple(leaf.shape)}')
        has_mapped = any(is_tensor(leaf) for leaf in leaves)
        input_dims = tree_unflatten([0 if is_tensor(leaf) else None for leaf in leaves], spec)

    if not params and not buffers and not has_mapped:
        return vmap(lambda _: stateless_call(template, {}, {}, args, kwargs, fn), randomness = randomness)(torch.empty(pop_size))

    return vmap(
        lambda p, b, ak: stateless_call(template, p, b, ak[0], ak[1], fn),
        in_dims = (0, 0, input_dims),
        randomness = randomness,
    )(params, buffers, (args, kwargs))

# live view into a single individual or subset

class PopulationView(Module):
    def __init__(self, population: Population, index: int | list[int]):
        super().__init__()
        object.__setattr__(self, '_pop', population)
        self._index = index

    @property
    def is_single(self) -> bool:
        return isinstance(self._index, int)

    @property
    def _indices(self) -> list[int]:
        return [self._index] if self.is_single else self._index

    def __len__(self) -> int:
        return 1 if self.is_single else len(self._index)

    @property
    def flat_params(self) -> Tensor:
        flat = self._pop.flat_params
        if not self._pop.is_population:
            return flat
        return flat[self._index]

    def unwrap(self) -> Module:
        return self._pop[self._index]

    def lerp_(self, other, weight: float | Tensor) -> PopulationView:
        self._pop.lerp_(other, weight, members = self._index if self._pop.is_population else None)
        return self

    def copy_from_(self, other, *, include_opt: bool = True, include_buffers: bool = True) -> PopulationView:
        if self._pop.is_population and self.is_single and getattr(other, 'pop_size', None) != self._pop.pop_size:
            self._pop.replace(self._index, other, reset_opt = not include_opt)
            return self
        self._pop.copy_from_(other, members = self._index if self._pop.is_population else None, include_opt = include_opt, include_buffers = include_buffers)
        return self

    def ema_(self, name: str, value: Tensor, decay: float | Tensor = 0.99) -> Tensor:
        buf = self._pop.ema_(name, value, decay = decay, members = self._index if self._pop.is_population else None)
        return buf[self._index] if (exists(buf) and self._pop.is_population and self._pop.population_parameters.store.is_populated(f'track/{name}', self._pop.pop_size)) else buf

    def tracked(self, name: str) -> Tensor | None:
        buf = self._pop.tracked(name)
        if not exists(buf) or not self._pop.is_population:
            return buf
        return buf[self._index] if self._pop.population_parameters.store.is_populated(f'track/{name}', self._pop.pop_size) else buf

    def reset_(self, **kwargs) -> PopulationView:
        self._pop.reset_(self._index if self._pop.is_population else None, **kwargs)
        return self

    def call(self, fn: str | Callable | None, *args, per_individual: bool | None = None, **kwargs):
        pop = self._pop
        if not pop.is_population:
            return invoke_module(pop.net, fn, args, kwargs)
        if not self.is_single:
            return pop.call(fn, *args, members = self._index, per_individual = per_individual, **kwargs)
        params, buffers = pop.population_parameters._member_tensors(self._index, track_grads = True)
        return stateless_call(pop.template, params, buffers, args, kwargs, fn)

    def forward(self, *args, per_individual: bool | None = None, **kwargs):
        return self.call(None, *args, per_individual = per_individual, **kwargs)

    def extra_repr(self) -> str:
        return f'index={self._index}' if self.is_single else f'indices={self._index}'

# main Population container

class Population(Module):
    def __init__(
        self,
        net: Module | Sequence[Module] | Callable[[], Module],
        pop_size: int | None = None,
        *,
        shared = (),
        init: Literal['reinit', 'clone'] | Callable[[Module], None] = 'reinit',
        noise_std: float = 0.,
        keep_first: bool | None = None,
        per_individual: bool = False,
        randomness: Literal['error', 'same', 'different'] = 'different',
    ):
        super().__init__()
        if exists(pop_size):
            check_pop_size(pop_size)

        self.per_individual = per_individual
        self.randomness = randomness
        self.net = None
        self.population_parameters = PopulationParameters()

        factory = None
        if callable(net) and not isinstance(net, Module):
            factory = net
            net = [net() for _ in range(pop_size)] if exists(pop_size) else net()

        if is_sequence(net):
            modules = list(net)
            if not modules or not all(isinstance(m, Module) for m in modules):
                raise ValueError('expected a non-empty sequence of nn.Module instances')
            if exists(pop_size) and pop_size != len(modules):
                raise ValueError(f'pop_size ({pop_size}) does not match number of networks ({len(modules)})')
            self._init_from_modules(modules, shared = shared, factory = factory)
            return

        if not isinstance(net, Module):
            raise TypeError(f'expected nn.Module, sequence, or callable factory, got {type(net).__name__}')

        self.net = net
        self.population_parameters.init_noop(net, factory = factory)
        if exists(pop_size):
            self.populate(pop_size, shared = shared, init = init, noise_std = noise_std, keep_first = keep_first)

    @classmethod
    def _from_parts(cls, pp: PopulationParameters, *, per_individual: bool, randomness: str, training: bool) -> Population:
        sub = cls.__new__(cls)
        Module.__init__(sub)
        sub.per_individual = per_individual
        sub.randomness = randomness
        sub.net = None
        sub.population_parameters = pp
        sub.train(training)
        return sub

    def _init_from_modules(self, modules: list[Module], *, shared = (), factory: Callable[[], Module] | None = None):
        self.net = None
        self.population_parameters.init_from_modules(modules, clone_template(modules[0]), shared = shared, factory = factory)
        self.train(modules[0].training)

    # properties

    @property
    def template(self) -> Module:
        return self.population_parameters.template

    @property
    def is_population(self) -> bool:
        return self.population_parameters.is_population

    @property
    def pop_size(self) -> int | None:
        return self.population_parameters.pop_size

    @property
    def params_dict(self) -> dict[str, Tensor]:
        return self.population_parameters.named_parameters_dict()

    @property
    def grads_dict(self) -> dict[str, Tensor | None]:
        return self.population_parameters.named_grads_dict()

    @property
    def buffers_dict(self) -> dict[str, Tensor]:
        return self.population_parameters.named_buffers_dict()

    @property
    def flat_params(self) -> Tensor:
        return self.population_parameters.flat_params

    @property
    def opt(self) -> PopulationOptimizer | None:
        return self.population_parameters._opt_owner()

    # optimizers & solo checkout

    def _resolve_opt(self, opt = None) -> PopulationOptimizer:
        pp = self.population_parameters
        if isinstance(opt, PopulationOptimizer):
            if opt.population is not self:
                raise ValueError('optimizer is not attached to this Population')
            return opt
        if isinstance(opt, str):
            return pp._optimizers[opt]
        if not pp._optimizers:
            raise ValueError('Population has no attached optimizer; call pop.optimizer(opt_cls, ...)')
        return pp._optimizers[pp._default_opt_name]

    def optimizer(self, opt_cls, *args, shared = False, shared_reduce = 'mean', per_individual: bool = False, name: str | None = None, **kwargs) -> PopulationOptimizer:
        return PopulationOptimizer(self, opt_cls, *args, shared = shared, shared_reduce = shared_reduce, per_individual = per_individual, name = name, **kwargs)

    def slot(self, name: str, *, opt = None) -> ParamSlot:
        pp = self.population_parameters
        spec = pp._schema.by_name.get(name)
        if not exists(spec):
            raise KeyError(f'no parameter named {name!r}')

        resolved = self._resolve_opt(opt) if (exists(opt) or pp._optimizers) else None
        if exists(resolved) and spec.requires_grad:
            return resolved.slot(name)
        return ParamSlot(name = name, param = self.params_dict[name], grad = self.grads_dict[name])

    def individual_dict(self, index: int = 0, *, opt = None, clone: bool = False) -> dict:
        return self.population_parameters.individual(index, opt = opt, clone = clone)

    def extract(self, index: int = 0, *, opt = None) -> tuple[Module, Optimizer]:
        return self._resolve_opt(opt).extract(index)

    def insert(self, index: int, module: Module, optimizer: Optimizer | None = None, *, opt = None) -> Population:
        self[index] = module
        if exists(optimizer):
            self._resolve_opt(opt).load_individual_state_(index, module, optimizer)
        return self

    @contextmanager
    def checkout(self, index: int = 0, *, opt = None) -> Iterator[tuple[Module, Optimizer]]:
        ind, ind_opt = self.extract(index, opt = opt)
        yield ind, ind_opt
        self.insert(index, ind, ind_opt, opt = opt)

    # lifecycle

    def reset_(self, selector = None, **kwargs) -> Population:
        self.population_parameters.reset_(selector, **kwargs)
        return self

    def replace(self, index: int, source, **kwargs) -> Population:
        self.population_parameters.replace_(index, source, **kwargs)
        return self

    def keep(self, members) -> Population:
        self.population_parameters.keep_(members)
        return self

    def delete(self, members) -> Population:
        self.population_parameters.delete_(members)
        return self

    def append(self, source) -> int:
        return self.population_parameters.append_(source)

    def zero_grad(self, set_to_none: bool = True, *, members = None):
        self.population_parameters.zero_grad_(members = members, set_to_none = set_to_none)

    def autograd(self, *losses: Tensor, params: PopulationParameters | None = None, **kwargs):
        return default(params, self.population_parameters).autograd(*losses, **kwargs)

    def surrogate(self, value: Tensor, grads: Tensor | Mapping[str, Tensor | None], *, params: PopulationParameters | None = None, names: Iterable[str] | None = None) -> Tensor:
        return default(params, self.population_parameters).surrogate(value, grads, names = names)

    def reconcile(self, *losses: Tensor, fn: Callable, params: PopulationParameters | None = None, **kwargs) -> Tensor:
        return default(params, self.population_parameters).reconcile(*losses, fn = fn, **kwargs)

    def set_grads_(self, grads: Tensor | Mapping[str, Tensor | None], *, accumulate: bool = False, names: Iterable[str] | None = None) -> Population:
        self.population_parameters.set_grads_(grads, accumulate = accumulate, names = names)
        return self

    def clone(self, *, include_opt: bool = True) -> Population:
        pp = self.population_parameters
        if include_opt:
            for opt in pp._optimizers.values():
                opt._sync_hp_from_groups()

        cloned_pp = pp.clone(include_opt = include_opt)
        sub = Population._from_parts(
            cloned_pp,
            per_individual = self.per_individual,
            randomness = self.randomness,
            training = self.training,
        )
        if not self.is_population:
            sub.net = cloned_pp._noop_net

        if include_opt:
            indices = list(range(pp.pop_size)) if self.is_population else None
            for opt in pp._optimizers.values():
                opt._clone_for_subpopulation(sub, indices)
            sub.population_parameters._default_opt_name = pp._default_opt_name
        return sub

    def lerp_(self, other, weight: float | Tensor, *, members = None) -> Population:
        self.population_parameters.lerp_(other, weight, members = members)
        return self

    def copy_from_(
        self,
        other,
        *,
        members = None,
        include_opt: bool = True,
        include_buffers: bool = True,
    ) -> Population:
        self.population_parameters.copy_from_(other, members = members, include_opt = include_opt, include_buffers = include_buffers)
        return self

    def ema_(self, name: str, value: Tensor, decay: float | Tensor = 0.99, *, members = None) -> Tensor:
        return self.population_parameters.ema_(name, value, decay = decay, members = members)

    def tracked(self, name: str) -> Tensor | None:
        return self.population_parameters.tracked(name)

    def clear_tracked(self, name: str | None = None, *, members = None) -> Population:
        self.population_parameters.clear_tracked(name, members = members)
        return self

    def _apply(self, fn, recurse = True):
        res = super()._apply(fn, recurse = recurse)
        pp = getattr(self, 'population_parameters', None)
        if exists(pp):
            for opt in pp._optimizers.values():
                opt._sync_optimizer_state()
        return res

    def populate(
        self,
        pop_size: int,
        *,
        shared = (),
        init: Literal['reinit', 'clone'] | Callable[[Module], None] = 'reinit',
        noise_std: float = 0.,
        keep_first: bool | None = None,
    ) -> Population:
        if self.is_population:
            raise ValueError(f'Population already populated (pop_size={self.pop_size})')
        check_pop_size(pop_size)
        if not callable(init) and init not in ('reinit', 'clone'):
            raise ValueError(f"init must be 'reinit', 'clone', or a callable, got {init!r}")
        if noise_std < 0.:
            raise ValueError(f'noise_std must be >= 0, got {noise_std}')

        preserve_first = default(keep_first, not callable(init))
        factory = self.population_parameters._factory
        ref = next(self.net.parameters(), next(self.net.buffers(), None))
        members = []

        for i in range(pop_size):
            if preserve_first and i == 0:
                members.append(deepcopy(self.net))
                continue
            if init == 'reinit' and exists(factory):
                member = factory()
                if exists(ref):
                    member = member.to(device = ref.device, dtype = ref.dtype if ref.is_floating_point() else None)
            else:
                member = deepcopy(self.net)
                init_module(member, init)
            if noise_std > 0.:
                add_param_noise_(member, noise_std)
            members.append(member)

        for opt in self.population_parameters._optimizers.values():
            opt._sync_noop_device()
        self._init_from_modules(members, shared = shared)
        self.population_parameters._on_populate_tracks_(pop_size, init = init, preserve_first = preserve_first)
        for opt in self.population_parameters._optimizers.values():
            opt._on_populate(pop_size, init = init, preserve_first = preserve_first)
        return self

    def unwrap(self, index: int | None = None) -> Module:
        if not self.is_population:
            if exists(index):
                self.population_parameters._normalize_index(index)
            return self.net
        if not exists(index) and self.pop_size != 1:
            raise ValueError(f'Cannot unwrap population of size {self.pop_size} without index')
        return self[default(index, 0)]

    def reduce(self, reduction = 'mean') -> Module:
        return self.population_parameters.reduce(reduction = reduction, training = self.training)

    def mean(self, weights: Tensor | Sequence[float] | None = None) -> Module:
        return self.population_parameters.mean(weights = weights, training = self.training)

    def train(self, mode: bool = True) -> Population:
        super().train(mode)
        if self.is_population:
            self.template.train(mode)
        return self

    # indexing & slicing

    def _slice_population(self, indices: list[int]) -> Population:
        pp = self.population_parameters
        for opt in pp._optimizers.values():
            opt._sync_hp_from_groups()

        if self.is_population:
            sub = Population._from_parts(
                pp.sliced(indices, clone_opts = True),
                per_individual = self.per_individual,
                randomness = self.randomness,
                training = self.training,
            )
        else:
            sub = Population(
                [deepcopy(self.net) for _ in indices],
                per_individual = self.per_individual,
                randomness = self.randomness,
            )

        for opt in pp._optimizers.values():
            opt._clone_for_subpopulation(sub, indices)
        sub.population_parameters._default_opt_name = pp._default_opt_name
        return sub

    def view(self, key) -> PopulationView:
        pp = self.population_parameters
        idx = pp._normalize_index(key) if is_int_index(key) else pp.resolve_members(key, non_empty = True)
        return PopulationView(self, idx)

    def __len__(self) -> int:
        return self.pop_size if self.is_population else 1

    def __iter__(self) -> Iterator[Module]:
        return (self[i] for i in range(len(self)))

    def __getitem__(self, key) -> Module:
        pp = self.population_parameters
        if not is_int_index(key):
            return self._slice_population(pp.resolve_members(key, non_empty = True))
        idx = pp._normalize_index(key)
        if not self.is_population:
            return self.net
        return clone_template(self.template, *pp._member_tensors(idx), training = self.training)

    def _copy_members_from_(self, dst_indices: list[int], src_pop: Population, src_indices: list[int], *, copy_opts: bool = False):
        pp, src_pp = self.population_parameters, src_pop.population_parameters
        if src_pp._schema.param_names != pp._schema.param_names or src_pp.buffer_names != pp.buffer_names:
            raise ValueError('source module parameter/buffer names do not match population')
        aliasing = src_pp is pp
        sources = [src_pp._member_tensors(src_i, clone = aliasing) for src_i in src_indices]
        with torch.no_grad():
            for dst_i, (src_p, src_b) in zip(dst_indices, sources):
                pp._copy_member_state_(dst_i, src_p, src_b, check_shapes = True)
            for prefix in ('track/', 'track_init/'):
                for key, dst in pp.store.items(prefix):
                    src = src_pp.store.get(f'{prefix}{key}')
                    if not exists(src) or not pp.store.is_populated(f'{prefix}{key}', pp.pop_size):
                        continue
                    src_rows = (src[src_indices] if src_pp.store.is_populated(f'{prefix}{key}', src_pp.pop_size) else src.unsqueeze(0).expand(len(src_indices), *src.shape)).clone()
                    dst[dst_indices] = src_rows.to(dst)
        if copy_opts:
            src_opts = src_pp._optimizers
            for name, opt in pp._optimizers.items():
                if name in src_opts:
                    opt._copy_from_other_opt_(dst_indices, src_opts[name], src_indices)

    def _set_single(self, idx: int, value):
        if isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], Module) and isinstance(value[1], Optimizer):
            self.insert(idx, value[0], value[1])
            return
        if isinstance(value, Mapping):
            self.population_parameters.load_individual_dict_(idx, value)
            return

        if isinstance(value, (Population, PopulationView)):
            if len(value) != 1:
                raise ValueError(f'Cannot assign {type(value).__name__} of size {len(value)} to a single slot')
            is_pop = isinstance(value, Population)
            self._copy_members_from_([idx], value if is_pop else value._pop, [0] if is_pop else value._indices, copy_opts = is_pop)
            return

        if not isinstance(value, Module):
            raise TypeError(f'expected nn.Module, (nn.Module, Optimizer), or dict, got {type(value).__name__}')

        src_params, src_buffers = dict(value.named_parameters()), dict(value.named_buffers())
        pp = self.population_parameters
        if list(src_params) != pp._schema.param_names or list(src_buffers) != pp.buffer_names:
            raise ValueError('source module parameter/buffer names do not match population')
        pp.load_individual_dict_(idx, {'named_parameters': src_params, 'named_buffers': src_buffers})

    def __setitem__(self, key, value):
        pp = self.population_parameters
        if is_int_index(key):
            self._set_single(pp._normalize_index(key), value)
            return

        indices = pp.resolve_members(key, non_empty = True)
        is_pop_or_view = isinstance(value, (Population, PopulationView))
        if not is_pop_or_view and not is_sequence(value):
            raise TypeError('slice assignment requires a Population, PopulationView, or sequence of modules')
        if len(value) != len(indices):
            raise ValueError(f'cannot assign {len(value)} items to {len(indices)} slots')

        if is_pop_or_view:
            is_pop = isinstance(value, Population)
            src_indices = list(range(len(value))) if is_pop else value._indices
            self._copy_members_from_(indices, value if is_pop else value._pop, src_indices, copy_opts = is_pop)
            return

        for dst, item in zip(indices, value):
            self._set_single(dst, item)

    # forward & arbitrary vectorized method dispatch

    def _call_flat(
        self,
        fn: str | Callable | None,
        flat: Tensor,
        args: tuple,
        kwargs: dict,
        *,
        members = None,
        per_individual: bool,
    ):
        if flat.ndim not in (1, 2):
            raise ValueError(f'flat params tensor must be 1D (D,) or 2D (K, D), got {tuple(flat.shape)}')

        pp = self.population_parameters
        unpacked = pp.unpack_flat(flat)
        named_p = pp.named_parameters_dict()
        named_b = pp.named_buffers_dict()

        if flat.ndim == 1:
            if exists(members):
                raise ValueError('members= requires 2D flat params (K, D)')
            full_p = {
                s.name: unpacked[s.name] if s.name in unpacked else (named_p[s.name][0] if (pp.is_population and not s.shared) else named_p[s.name])
                for s in pp._schema.specs
            }
            full_b = {
                name: (buf[0] if (pp.is_population and not pp.is_shared_buffer(name)) else buf)
                for name, buf in named_b.items()
            }
            return stateless_call(self.template, full_p, full_b, args, kwargs, fn)

        if exists(members) and not pp.is_population:
            raise ValueError('members= requires a populated Population')
        indices = pp.resolve_members(members) if exists(members) else None
        if exists(indices):
            if flat.shape[0] == pp.pop_size:
                unpacked = {k: v[indices] for k, v in unpacked.items()}
            num_members = len(indices)
        else:
            num_members = flat.shape[0]

        def adapt(t: Tensor, is_pop_t: bool, *, mutable: bool = False) -> Tensor:
            if not is_pop_t:
                out = t.unsqueeze(0).expand(num_members, *t.shape)
                return out.clone() if mutable else out
            if exists(indices):
                return t[indices]
            out = t if t.shape[0] == num_members else t[:1].expand(num_members, *t.shape[1:])
            return out.clone() if (mutable and t.shape[0] != num_members) else out

        full_p = {
            s.name: unpacked[s.name] if s.name in unpacked else adapt(named_p[s.name], pp.is_population and not s.shared)
            for s in pp._schema.specs
        }
        full_b = {
            name: adapt(buf, pp.is_population and not pp.is_shared_buffer(name), mutable = self.training)
            for name, buf in named_b.items()
        }
        return vmap_forward(
            self.template, full_p, full_b, num_members, args, kwargs,
            fn = fn, per_individual = per_individual, randomness = self.randomness,
        )

    def call(
        self,
        fn: str | Callable | None,
        *args,
        params: PopulationParameters | Mapping[str, Tensor] | Tensor | None = None,
        members = None,
        per_individual: bool | None = None,
        **kwargs,
    ):
        per_individual = default(per_individual, self.per_individual)

        if is_tensor(params):
            return self._call_flat(fn, params, args, kwargs, members = members, per_individual = per_individual)

        if isinstance(params, Mapping):
            first = next(iter(params.values()), None)
            num_members = first.shape[0] if exists(first) else (self.pop_size or 1)
            buffers = {}
            if self.is_population:
                pp = self.population_parameters
                buffers = {
                    name: (buf.unsqueeze(0).expand(num_members, *buf.shape) if pp.is_shared_buffer(name) else buf)
                    for name, buf in pp.named_buffers_dict().items()
                }
            return vmap_forward(
                self.template, dict(params), buffers, num_members, args, kwargs,
                fn = fn, per_individual = per_individual, randomness = self.randomness,
            )

        if exists(params) and not isinstance(params, PopulationParameters):
            raise TypeError(f'params must be PopulationParameters, Mapping, Tensor, or None, got {type(params)}')

        pp = default(params, self.population_parameters)
        if not pp.is_population:
            if exists(members):
                raise ValueError('members= requires a populated Population')
            if exists(params):
                return stateless_call(self.template, pp.named_parameters_dict(), pp.named_buffers_dict(), args, kwargs, fn)
            return invoke_module(self.net, fn, args, kwargs)

        indices = pp.resolve_members(members) if exists(members) else None
        num_members = len(indices) if exists(indices) else pp.pop_size
        sel = default(indices, slice(None))

        orig_buffers = pp.named_buffers_dict()
        shared_mut_buffers = bool(pp._schema.shared_buffers) and self.training
        if exists(indices) or pp._schema.shared_buffers:
            buffers = {
                name: (
                    (buf.unsqueeze(0).expand(num_members, *buf.shape).clone() if shared_mut_buffers else buf.unsqueeze(0).expand(num_members, *buf.shape))
                    if pp.is_shared_buffer(name) else buf[sel]
                )
                for name, buf in orig_buffers.items()
            }
        else:
            buffers = orig_buffers

        named = pp.named_parameters_dict(track_grads = not exists(params))
        if exists(indices) or pp._schema.has_shared:
            named = {
                s.name: (named[s.name].unsqueeze(0).expand(num_members, *s.shape) if s.shared else named[s.name][sel])
                for s in pp._schema.specs
            }

        out = vmap_forward(
            self.template, named, buffers, num_members, args, kwargs,
            fn = fn, per_individual = per_individual, randomness = self.randomness,
        )

        # write back any in-place shared buffer updates (e.g. BatchNorm running stats)
        if shared_mut_buffers and not exists(params):
            with torch.no_grad():
                for name in pp._schema.shared_buffers:
                    dst = orig_buffers[name]
                    dst.copy_(reduce_pop(buffers[name], 'mean' if dst.is_floating_point() else 'first'))

        return out

    def forward(
        self,
        *args,
        params: PopulationParameters | Mapping[str, Tensor] | Tensor | None = None,
        members = None,
        per_individual: bool | None = None,
        **kwargs,
    ):
        return self.call(None, *args, params = params, members = members, per_individual = per_individual, **kwargs)

    def extra_repr(self) -> str:
        return 'noop' if not self.is_population else f'pop_size={self.pop_size}, randomness={self.randomness!r}'
