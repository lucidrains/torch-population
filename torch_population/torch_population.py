from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
from typing import Literal

import torch
from torch import Tensor, is_tensor
from torch.func import functional_call, vmap
from torch.nn import Module
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
)

# helpers

def _check_pop_size(pop_size: int):
    if isinstance(pop_size, bool) or not isinstance(pop_size, int) or pop_size < 1:
        raise ValueError(f'pop_size must be a positive integer, got {pop_size!r}')

@torch.no_grad()
def _add_param_noise_(module: Module, noise_std: float):
    for p in module.parameters():
        if p.is_floating_point():
            p.add_(torch.randn_like(p) * noise_std)

def _vmap_call(template: Module, params: dict[str, Tensor], buffers: dict[str, Tensor], pop_size: int, args: tuple, kwargs: dict, *, per_individual: bool = False, randomness: str = 'different'):
    leaves, spec = tree_flatten((args, kwargs)) if per_individual else ([], None)
    tensor_pos = [i for i, leaf in enumerate(leaves) if is_tensor(leaf)]
    mapped = [leaves[i] for i in tensor_pos]

    for t in mapped:
        if t.ndim == 0 or t.shape[0] != pop_size:
            raise ValueError(f'per_individual=True expects leading dim {pop_size}, got {tuple(t.shape)}')

    if not params and not buffers and not mapped:
        return vmap(lambda _: functional_call(template, ({}, {}), args, kwargs), randomness = randomness)(torch.empty(pop_size))

    def call_member(mp, mb, *minputs):
        if not tensor_pos:
            return functional_call(template, (mp, mb), args, kwargs)
        merged = list(leaves)
        for pos, t in zip(tensor_pos, minputs):
            merged[pos] = t
        m_args, m_kwargs = tree_unflatten(merged, spec)
        return functional_call(template, (mp, mb), m_args, m_kwargs)

    return vmap(call_member, in_dims = (0, 0, *(0 for _ in mapped)), randomness = randomness)(params, buffers, *mapped)

# live view into a single individual or subset

class PopulationView(Module):
    def __init__(self, population: Population, index: int | list[int]):
        super().__init__()
        object.__setattr__(self, '_pop', population)
        self._index = index

    @property
    def is_single(self) -> bool:
        return isinstance(self._index, int)

    def __len__(self) -> int:
        return 1 if self.is_single else len(self._index)

    def unwrap(self) -> Module:
        return self._pop[self._index]

    def forward(self, *args, per_individual: bool = False, **kwargs):
        pop = self._pop
        if not pop.is_population:
            return pop.net(*args, **kwargs)
        if not self.is_single:
            return pop(*args, members = self._index, per_individual = per_individual, **kwargs)
        state = pop.population_parameters._member_tensors(self._index, track_grads = True)
        return functional_call(pop.template, state, args, kwargs)

    def extra_repr(self) -> str:
        return f'index={self._index}' if self.is_single else f'indices={self._index}'

# main Population container

class Population(Module):
    def __init__(self, net: Module | Sequence[Module] | Callable[[], Module], pop_size: int | None = None, *, shared = (), init: Literal['reinit', 'clone'] | Callable[[Module], None] = 'reinit', noise_std: float = 0., keep_first: bool | None = None, randomness: Literal['error', 'same', 'different'] = 'different'):
        super().__init__()
        if exists(pop_size):
            _check_pop_size(pop_size)

        self.randomness = randomness
        self.net = None
        self.population_parameters = PopulationParameters()

        if callable(net) and not isinstance(net, Module):
            net = [net() for _ in range(pop_size)] if exists(pop_size) else net()

        if is_sequence(net):
            modules = list(net)
            if not modules or not all(isinstance(m, Module) for m in modules):
                raise ValueError('expected a non-empty sequence of nn.Module instances')
            if exists(pop_size) and pop_size != len(modules):
                raise ValueError(f'pop_size ({pop_size}) does not match number of networks ({len(modules)})')
            self._init_from_modules(modules, shared = shared)
            return

        if not isinstance(net, Module):
            raise TypeError(f'expected nn.Module, sequence, or callable factory, got {type(net).__name__}')

        self.net = net
        self.population_parameters.init_noop(net)
        if exists(pop_size):
            self.populate(pop_size, shared = shared, init = init, noise_std = noise_std, keep_first = keep_first)

    @classmethod
    def _from_parts(cls, pp: PopulationParameters, randomness: str, training: bool) -> Population:
        sub = cls.__new__(cls)
        Module.__init__(sub)
        sub.randomness = randomness
        sub.net = None
        sub.population_parameters = pp
        sub.train(training)
        return sub

    def _init_from_modules(self, modules: list[Module], *, shared = ()):
        self.net = None
        self.population_parameters.init_from_modules(modules, clone_template(modules[0]), shared = shared)
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
    def opt(self) -> PopulationOptimizer | None:
        pp = self.population_parameters
        return pp._optimizers.get(pp._default_opt_name) if exists(pp._default_opt_name) else None

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

    def optimizer(self, opt_cls, *args, shared = False, per_individual: bool = False, name = None, **kwargs):
        return PopulationOptimizer(self, opt_cls, *args, shared = shared, per_individual = per_individual, name = name, **kwargs)

    def slot(self, name: str, *, opt = None) -> ParamSlot:
        pp = self.population_parameters
        params = self.params_dict
        if name not in params:
            raise KeyError(f'no parameter named {name!r}')
        resolved = self._resolve_opt(opt) if (exists(opt) or pp._optimizers) else None
        spec = pp._schema.by_name.get(name)
        if exists(resolved) and exists(spec) and spec.requires_grad:
            return resolved.slot(name)
        return ParamSlot(name = name, param = params[name], grad = self.grads_dict[name])

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

    def _apply(self, fn, recurse = True):
        res = super()._apply(fn, recurse = recurse)
        pp = getattr(self, 'population_parameters', None)
        if exists(pp):
            for opt in pp._optimizers.values():
                opt._sync_optimizer_state()
        return res

    def populate(self, pop_size: int, *, shared = (), init: Literal['reinit', 'clone'] | Callable[[Module], None] = 'reinit', noise_std: float = 0., keep_first: bool | None = None) -> Population:
        if self.is_population:
            raise ValueError(f'Population already populated (pop_size={self.pop_size})')
        _check_pop_size(pop_size)
        if not callable(init) and init not in ('reinit', 'clone'):
            raise ValueError(f"init must be 'reinit', 'clone', or a callable, got {init!r}")
        if noise_std < 0.:
            raise ValueError(f'noise_std must be >= 0, got {noise_std}')

        preserve_first = default(keep_first, not callable(init))
        members = []
        for i in range(pop_size):
            m = deepcopy(self.net)
            members.append(m)
            if preserve_first and i == 0:
                continue
            init_module(m, init)
            if noise_std > 0.:
                _add_param_noise_(m, noise_std)

        self._init_from_modules(members, shared = shared)
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

    def mean(self, weights: Tensor | Sequence[float] | None = None) -> Module:
        if not self.is_population:
            return deepcopy(self.net)
        avg = self.population_parameters.mean_dict(weights = weights)
        return clone_template(self.template, avg['named_parameters'], avg['named_buffers'], training = self.training)

    def train(self, mode: bool = True) -> Population:
        super().train(mode)
        if self.is_population:
            self.template.train(mode)
        return self

    # indexing & slicing

    def _slice_population(self, indices: list[int]) -> Population:
        pp = self.population_parameters
        for opt in pp._optimizers.values():
            opt._resolve_kwargs()
        if self.is_population:
            sub = Population._from_parts(pp.sliced(indices, clone_opts = True), self.randomness, self.training)
        else:
            sub = Population([deepcopy(self.net) for _ in indices], randomness = self.randomness)
        for opt in pp._optimizers.values():
            opt._clone_for_subpopulation(sub, indices)
        sub.population_parameters._default_opt_name = pp._default_opt_name
        return sub

    def _copy_opts_from_(self, src_pop: Population, dst_idx: list[int], src_idx: list[int]):
        src_opts = src_pop.population_parameters._optimizers
        for name, opt in self.population_parameters._optimizers.items():
            if name in src_opts:
                opt._copy_from_other_opt_(dst_idx, src_opts[name], src_idx)

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

    def _set_single(self, idx: int, value):
        if isinstance(value, tuple) and len(value) == 2:
            mod, opt = value
            if isinstance(mod, Module) and isinstance(opt, Optimizer):
                self.insert(idx, mod, opt)
                return
        if isinstance(value, Mapping):
            self.population_parameters.load_individual_dict_(idx, value)
            return

        src_pop = value if isinstance(value, Population) else None
        mod = value.unwrap() if isinstance(value, (Population, PopulationView)) else value
        if not isinstance(mod, Module):
            raise TypeError(f'expected nn.Module, (nn.Module, Optimizer), or dict, got {type(value).__name__}')

        src_p, src_b = dict(mod.named_parameters()), dict(mod.named_buffers())
        pp = self.population_parameters
        if list(src_p) != [s.name for s in pp._schema.specs] or list(src_b) != pp.buffer_names:
            raise ValueError('source module parameter/buffer names do not match population')

        pp.load_individual_dict_(idx, {'named_parameters': src_p, 'named_buffers': src_b})
        if exists(src_pop):
            self._copy_opts_from_(src_pop, [idx], [0])

    def __setitem__(self, key, value):
        pp = self.population_parameters
        if is_int_index(key):
            self._set_single(pp._normalize_index(key), value)
            return
        indices = pp.resolve_members(key, non_empty = True)
        is_pop = isinstance(value, Population)
        if not is_pop and not is_sequence(value):
            raise TypeError('slice assignment requires a Population or sequence of modules')
        if len(value) != len(indices):
            raise ValueError(f'cannot assign {len(value)} items to {len(indices)} slots')
        for dst, item in zip(indices, value):
            self._set_single(dst, item)
        if is_pop:
            self._copy_opts_from_(value, indices, list(range(len(value))))

    # forward

    def forward(self, *args, params: PopulationParameters | Mapping[str, Tensor] | None = None, members = None, per_individual: bool = False, **kwargs):
        if isinstance(params, Mapping):
            first = next(iter(params.values()), None)
            num_members = first.shape[0] if exists(first) else (self.pop_size or 1)
            buffers = self.buffers_dict if self.is_population else {}
            return _vmap_call(self.template, dict(params), buffers, num_members, args, kwargs, per_individual = per_individual, randomness = self.randomness)

        if exists(params) and not isinstance(params, PopulationParameters):
            raise TypeError(f'params must be PopulationParameters, Mapping, or None, got {type(params)}')

        pp = default(params, self.population_parameters)
        if not pp.is_population:
            if exists(members) and not exists(params):
                raise ValueError('members= requires a populated Population')
            if exists(params):
                return functional_call(self.template, (pp.named_parameters_dict(), pp.named_buffers_dict()), args, kwargs)
            return self.net(*args, **kwargs)

        idx = pp.resolve_members(members) if exists(members) else None
        buffers = pp.named_buffers_dict()
        if exists(idx):
            buffers = {k: t[idx] for k, t in buffers.items()}

        named = pp.named_parameters_dict(track_grads = not exists(params))
        num_members = len(idx) if exists(idx) else pp.pop_size
        if exists(idx) or pp._schema.has_shared:
            sl = idx if exists(idx) else slice(None)
            batched = {
                s.name: (named[s.name].unsqueeze(0).expand(num_members, *s.shape) if s.shared else named[s.name][sl])
                for s in pp._schema.specs
            }
        else:
            batched = named

        return _vmap_call(self.template, batched, buffers, num_members, args, kwargs, per_individual = per_individual, randomness = self.randomness)

    def extra_repr(self) -> str:
        return 'noop' if not self.is_population else f'pop_size={self.pop_size}, randomness={self.randomness!r}'
