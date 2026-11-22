from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
from typing import Literal

import torch
from torch import Tensor, is_tensor
from torch.func import functional_call, vmap
from torch.nn import Module, Parameter
from torch.optim import Optimizer
from torch.utils._pytree import tree_flatten, tree_unflatten

from torch_population.optimizers import SGD, Adam, AdamW, Muon, PopulationOptimizer
from torch_population.parameters import PopulationParameters
from torch_population.store import (
    ParamSlot,
    default,
    exists,
    is_int_index,
    materialize_from_template,
    reset_module_parameters,
    slice_pop,
)

# helpers

def _to_meta_preserving_ties(module: Module) -> Module:
    template = deepcopy(module)
    memo: dict[int, Tensor] = {}

    for submod in template.modules():
        for name, param in list(submod._parameters.items()):
            if exists(param):
                if id(param) not in memo:
                    memo[id(param)] = Parameter(param.detach().to('meta'), requires_grad = param.requires_grad)
                submod._parameters[name] = memo[id(param)]
        for name, buf in list(submod._buffers.items()):
            if exists(buf):
                if id(buf) not in memo:
                    memo[id(buf)] = buf.detach().to('meta')
                submod._buffers[name] = memo[id(buf)]

    return template

def _vmap_functional_call(
    template: Module,
    params: dict[str, Tensor],
    buffers: dict[str, Tensor],
    pop_size: int,
    args: tuple,
    kwargs: dict,
    *,
    per_individual: bool = False,
    randomness: str = 'different',
):
    if not per_individual:
        if params or buffers:
            fn = lambda p, b: functional_call(template, (p, b), args, kwargs)
            return vmap(fn, in_dims = (0, 0), randomness = randomness)(params, buffers)
        fn = lambda _: functional_call(template, ({}, {}), args, kwargs)
        return vmap(fn, in_dims = 0, randomness = randomness)(torch.empty(pop_size))

    leaves, spec = tree_flatten((args, kwargs))
    tensor_pos = [i for i, leaf in enumerate(leaves) if is_tensor(leaf)]
    tensor_leaves = [leaves[i] for i in tensor_pos]

    for t in tensor_leaves:
        if t.ndim == 0 or t.shape[0] != pop_size:
            raise ValueError(f'per_individual=True expects tensor inputs with leading dim {pop_size}, got shape {tuple(t.shape)}')

    if not params and not buffers and not tensor_leaves:
        fn = lambda _: functional_call(template, ({}, {}), args, kwargs)
        return vmap(fn, in_dims = 0, randomness = randomness)(torch.empty(pop_size))

    def call_per_ind(p, b, *mapped):
        merged = list(leaves)
        for pos, t in zip(tensor_pos, mapped):
            merged[pos] = t
        call_args, call_kwargs = tree_unflatten(merged, spec)
        return functional_call(template, (p, b), call_args, call_kwargs)

    in_dims = (0, 0, *(0 for _ in tensor_leaves))
    return vmap(call_per_ind, in_dims = in_dims, randomness = randomness)(params, buffers, *tensor_leaves)

# live view into a single individual or subset of a Population

class PopulationView(Module):
    def __init__(self, population: Population, index: int | list[int]):
        super().__init__()
        self._pop_ref = [population]
        self._index = index

    @property
    def _pop(self) -> Population:
        return self._pop_ref[0]

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

        pp = pop.population_parameters
        idx = self._index
        params = pp.named_parameters_dict(track_grads = True)
        buffers = pp.named_buffers_dict()
        member_p = {s.name: (params[s.name] if s.shared else params[s.name][idx]) for s in pp.specs}
        member_b = {k: b[idx] for k, b in buffers.items()}
        return functional_call(pop._template, (member_p, member_b), args, kwargs)

    def extra_repr(self) -> str:
        return f'index={self._index}' if self.is_single else f'indices={self._index}'

# main class

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
        randomness: Literal['error', 'same', 'different'] = 'different',
    ):
        super().__init__()
        if exists(pop_size) and (isinstance(pop_size, bool) or not isinstance(pop_size, int) or pop_size < 1):
            raise ValueError(f'pop_size must be a positive integer or None, got {pop_size!r}')

        self.randomness = randomness
        self._template_ref: list[Module] = []
        self.net: Module | None = None
        self.population_parameters = PopulationParameters()

        if isinstance(net, Sequence) and not isinstance(net, (str, bytes)):
            modules = list(net)
            if not modules:
                raise ValueError('expected at least one network in sequence')
            if not all(isinstance(m, Module) for m in modules):
                raise TypeError('all items in sequence must be torch.nn.Module instances')
            if exists(pop_size) and pop_size != len(modules):
                raise ValueError(f'pop_size ({pop_size}) does not match number of networks ({len(modules)})')
            self._init_from_modules(modules, shared = shared)
            return

        if not isinstance(net, Module) and callable(net):
            if not exists(pop_size):
                self._init_noop(net())
            else:
                self._init_from_modules([net() for _ in range(pop_size)], shared = shared)
            return

        if not isinstance(net, Module):
            raise TypeError(f'expected nn.Module, sequence of nn.Module, or callable factory, got {type(net).__name__}')

        self._init_noop(net)
        if exists(pop_size):
            self.populate(pop_size, shared = shared, init = init, noise_std = noise_std, keep_first = keep_first)

    def _init_noop(self, net: Module):
        self.net = net
        self._template_ref = []
        self.population_parameters.init_noop(net)

    def _init_from_modules(self, modules: list[Module], *, shared = ()):
        ref = modules[0]
        template = _to_meta_preserving_ties(ref)
        self.net = None
        self._template_ref = [template]
        self.population_parameters.init_from_modules(modules, template, shared = shared)
        self.train(ref.training)

    @property
    def _template(self) -> Module:
        return self._template_ref[0]

    @property
    def template(self) -> Module:
        return self.net if not self.is_population else self._template_ref[0]

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

    @property
    def optimizers(self) -> dict[str, PopulationOptimizer]:
        return dict(self.population_parameters._optimizers)

    def _register_optimizer(self, name: str, opt: PopulationOptimizer):
        pp = self.population_parameters
        pp._optimizers[name] = opt
        pp._opt_owners[name] = [opt]
        pp._default_opt_name = name

    def _resolve_optimizer(self, opt: str | PopulationOptimizer | None = None) -> PopulationOptimizer:
        pp = self.population_parameters
        if isinstance(opt, PopulationOptimizer):
            if opt.population is not self:
                raise ValueError('optimizer is not attached to this Population')
            return opt
        if isinstance(opt, str):
            if opt not in pp._optimizers:
                raise KeyError(f'no optimizer named {opt!r} attached to this Population')
            return pp._optimizers[opt]
        if not pp._optimizers:
            raise ValueError('Population has no attached optimizer; attach one with pop.adam(), pop.adamw(), pop.sgd(), or pop.muon()')
        return pp._optimizers[pp._default_opt_name]

    def adam(self, *args, **kwargs) -> Adam:
        return Adam(self, *args, **kwargs)

    def adamw(self, *args, **kwargs) -> AdamW:
        return AdamW(self, *args, **kwargs)

    def sgd(self, *args, **kwargs) -> SGD:
        return SGD(self, *args, **kwargs)

    def muon(self, *args, **kwargs) -> Muon:
        return Muon(self, *args, **kwargs)

    def slot(self, name: str, *, opt: str | PopulationOptimizer | None = None) -> ParamSlot:
        pp = self.population_parameters
        params = pp.named_parameters_dict()
        if name not in params:
            raise KeyError(f'no parameter named {name!r}')
        resolved = self._resolve_optimizer(opt) if (exists(opt) or pp._optimizers) else None
        if exists(resolved) and name in resolved._param_names:
            return resolved.slot(name)
        return ParamSlot(name = name, param = params[name], grad = pp.named_grads_dict()[name])

    def individual_dict(self, index: int = 0, *, opt: str | PopulationOptimizer | None = None, clone: bool = False) -> dict:
        return self.population_parameters.individual(index, opt = opt, clone = clone)

    def extract(self, index: int = 0, *, opt: str | PopulationOptimizer | None = None) -> tuple[Module, Optimizer]:
        return self._resolve_optimizer(opt).extract(index)

    def insert(
        self,
        index: int,
        module: Module,
        optimizer: Optimizer | None = None,
        *,
        opt: str | PopulationOptimizer | None = None,
    ) -> Population:
        self[index] = module
        if exists(optimizer):
            self._resolve_optimizer(opt).load_individual_state_(index, module, optimizer)
        return self

    @contextmanager
    def checkout(self, index: int = 0, *, opt: str | PopulationOptimizer | None = None) -> Iterator[tuple[Module, Optimizer]]:
        ind, ind_opt = self.extract(index, opt = opt)
        yield ind, ind_opt
        self.insert(index, ind, ind_opt, opt = opt)

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

    def zero_grad(self, set_to_none: bool = True):
        super().zero_grad(set_to_none = set_to_none)
        self.population_parameters.clear_grad_tracking()

    def _apply(self, fn, recurse = True):
        out = super()._apply(fn, recurse = recurse)
        pp = getattr(self, 'population_parameters', None)
        if exists(pp):
            for attached in pp._optimizers.values():
                attached._sync_optimizer_state()
        return out

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
            raise ValueError(f'Population is already populated with pop_size={self.pop_size}; use Population(pop[0], {pop_size})')
        if isinstance(pop_size, bool) or not isinstance(pop_size, int) or pop_size < 1:
            raise ValueError(f'pop_size must be a positive integer or None, got {pop_size!r}')
        if not callable(init) and init not in ('reinit', 'clone'):
            raise ValueError(f"init must be 'reinit', 'clone', or a callable, got {init!r}")
        if noise_std < 0.:
            raise ValueError(f'noise_std must be >= 0, got {noise_std}')

        preserve_first = default(keep_first, not callable(init))
        base = self.net
        members: list[Module] = []

        for idx in range(pop_size):
            member = deepcopy(base)
            if not (preserve_first and idx == 0):
                if callable(init):
                    init(member)
                elif init == 'reinit':
                    reset_module_parameters(member)
                if noise_std > 0.:
                    with torch.no_grad():
                        for p in member.parameters():
                            if p.is_floating_point():
                                p.add_(torch.randn_like(p) * noise_std)
            members.append(member)

        self._init_from_modules(members, shared = shared)
        for attached in self.population_parameters._optimizers.values():
            attached._on_populate(pop_size, init = init, preserve_first = preserve_first)
        return self

    def unwrap(self, index: int | None = None) -> Module:
        if not self.is_population:
            if exists(index):
                self._normalize_index(index)
            return self.net
        if not exists(index):
            if self.pop_size == 1:
                return self[0]
            raise ValueError(f'Cannot unwrap a population of size {self.pop_size} without an index; use pop[i] or pop.unwrap(i)')
        return self[index]

    def mean(self, weights: Tensor | Sequence[float] | None = None) -> Module:
        if not self.is_population:
            return deepcopy(self.net)
        reduced = self.population_parameters.mean_dict(weights = weights)
        return materialize_from_template(self._template, reduced['named_parameters'], reduced['named_buffers'], self.training)

    def train(self, mode: bool = True) -> Population:
        super().train(mode)
        if self._template_ref:
            self._template_ref[0].train(mode)
        return self

    def _normalize_index(self, index: int | Tensor) -> int:
        return self.population_parameters._normalize_index(index)

    def _normalize_indices(self, key) -> list[int]:
        indices = self.population_parameters.resolve_members(key)
        if not indices:
            raise ValueError('cannot index an empty population')
        return indices

    def _slice_population(self, indices: list[int]) -> Population:
        sub = Population.__new__(Population)
        Module.__init__(sub)
        sub.randomness = self.randomness
        sub.net = None
        sub._template_ref = []
        sub.population_parameters = PopulationParameters()

        pp = self.population_parameters
        if not self.is_population:
            sub._init_from_modules([deepcopy(self.net) for _ in indices])
        else:
            template = deepcopy(self._template)
            sub._template_ref = [template]
            frozen_store = pp._frozen_store.clone_store()
            frozen_store.keep_(indices, pp.pop_size)
            buffer_store = pp._buffer_store.clone_store()
            buffer_store.keep_(indices, pp.pop_size)

            sub.population_parameters._init_from_sliced(
                pop_size = len(indices),
                template = template,
                specs = pp.specs,
                sd_map = pp._state_dict_key_map,
                trainable_flats = [Parameter(slice_pop(f, indices), requires_grad = True) for f in pp._trainable_flats],
                shared_trainable_flats = [Parameter(f.detach().clone(), requires_grad = True) for f in pp._shared_trainable_flats],
                frozen_store = frozen_store,
                buffer_store = buffer_store,
            )
            sub.train(self.training)

        for attached in pp._optimizers.values():
            attached._clone_for_subpopulation(sub, indices)
        sub.population_parameters._default_opt_name = pp._default_opt_name
        return sub

    def view(self, key) -> PopulationView:
        idx = self._normalize_index(key) if is_int_index(key) else self._normalize_indices(key)
        return PopulationView(self, idx)

    def __len__(self) -> int:
        return self.pop_size if self.is_population else 1

    def __iter__(self) -> Iterator[Module]:
        for idx in range(len(self)):
            yield self[idx]

    def __getitem__(self, key) -> Module:
        if is_int_index(key):
            idx = self._normalize_index(key)
            if not self.is_population:
                return self.net
            ind = self.population_parameters.individual(idx, clone = False)
            return materialize_from_template(self._template, ind['named_parameters'], ind['named_buffers'], self.training)
        return self._slice_population(self._normalize_indices(key))

    def __setitem__(self, key, value):
        if is_int_index(key):
            idx = self._normalize_index(key)
            if isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], Module) and isinstance(value[1], Optimizer):
                self.insert(idx, value[0], value[1])
                return
            if isinstance(value, Mapping):
                self.population_parameters.load_individual_dict_(idx, value)
                return

            src_pop = value if isinstance(value, Population) else None
            if isinstance(value, (Population, PopulationView)):
                value = value.unwrap()
            if not isinstance(value, Module):
                raise TypeError(f'expected nn.Module, (nn.Module, Optimizer), or dict, got {type(value).__name__}')

            src_p = dict(value.named_parameters(remove_duplicate = True))
            src_b = dict(value.named_buffers(remove_duplicate = True))
            pp = self.population_parameters
            if list(src_p) != pp.param_names:
                raise ValueError('source module parameter names do not match population')
            if list(src_b) != pp.buffer_names:
                raise ValueError('source module buffer names do not match population')

            pp.load_individual_dict_(idx, {'named_parameters': src_p, 'named_buffers': src_b})
            if exists(src_pop):
                src_opts = src_pop.population_parameters._optimizers
                for opt_name, attached in pp._optimizers.items():
                    if opt_name in src_opts:
                        attached._copy_from_other_opt_([idx], src_opts[opt_name], [0])
            return

        indices = self._normalize_indices(key)
        if isinstance(value, Population) and value.is_population:
            if len(value) != len(indices):
                raise ValueError(f'cannot assign population of size {len(value)} to {len(indices)} slots')
            for di, si in zip(indices, range(len(value))):
                self[di] = value[si]
            src_opts = value.population_parameters._optimizers
            for opt_name, attached in self.population_parameters._optimizers.items():
                if opt_name in src_opts:
                    attached._copy_from_other_opt_(indices, src_opts[opt_name], list(range(len(value))))
            return

        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            if len(value) != len(indices):
                raise ValueError(f'cannot assign {len(value)} items to {len(indices)} slots')
            for di, item in zip(indices, value):
                self[di] = item
            return

        raise TypeError('slice assignment requires a Population or sequence of nn.Module / (nn.Module, Optimizer)')

    def forward(
        self,
        *args,
        params: PopulationParameters | Mapping[str, Tensor] | None = None,
        members = None,
        per_individual: bool = False,
        **kwargs,
    ):
        if isinstance(params, Mapping):
            first_t = next(iter(params.values()), None)
            p_size = first_t.shape[0] if exists(first_t) else (self.pop_size or 1)
            buffers = self.population_parameters.named_buffers_dict() if self.is_population else {}
            return _vmap_functional_call(
                self.template, dict(params), buffers, p_size, args, kwargs,
                per_individual = per_individual, randomness = self.randomness,
            )

        if exists(params) and not isinstance(params, PopulationParameters):
            raise TypeError(f'params must be PopulationParameters, Mapping, or None, got {type(params).__name__}')

        pp = default(params, self.population_parameters)
        if not pp.is_population:
            if exists(members) and not exists(params):
                raise ValueError('members= requires a populated Population')
            if exists(params):
                return functional_call(self.template, (pp.named_parameters_dict(), pp.named_buffers_dict()), args, kwargs)
            return self.net(*args, **kwargs)

        idx = pp.resolve_members(members) if exists(members) else None
        n = len(idx) if exists(idx) else pp.pop_size
        return _vmap_functional_call(
            self.template,
            pp.batched_parameters_dict(members = members, track_grads = not exists(params)),
            pp.batched_buffers_dict(members = members),
            n, args, kwargs,
            per_individual = per_individual, randomness = self.randomness,
        )

    def extra_repr(self) -> str:
        if not self.is_population:
            return 'noop'
        return f'pop_size={self.pop_size}, randomness={self.randomness!r}'
