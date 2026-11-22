from copy import deepcopy
from typing import ClassVar

import pytest
import torch
from torch import nn

from torch_population import (
    SGD,
    Muon,
    Population,
    PopulationOptimizer,
    newton_schulz,
)

param = pytest.mark.parametrize


def make_mlp():
    return nn.Sequential(nn.Linear(8, 16), nn.SiLU(), nn.Linear(16, 4))


# 1. SGD (with momentum, dampening, Nesterov, shared, extract/writeback, functional_step)

@param('momentum', [0., 0.9])
@param('nesterov', [False, True])
def test_sgd_matches_torch_sgd_noop_and_population(momentum, nesterov):
    if nesterov and momentum == 0.:
        pytest.skip('nesterov requires momentum > 0')

    torch.manual_seed(0)

    # noop mode exact parity
    net_a = make_mlp()
    net_b = deepcopy(net_a)
    pop_noop = Population(net_a)
    opt_a = pop_noop.sgd(lr = 0.05, momentum = momentum, weight_decay = 1e-2, nesterov = nesterov)
    opt_b = torch.optim.SGD(net_b.parameters(), lr = 0.05, momentum = momentum, weight_decay = 1e-2, nesterov = nesterov)

    for _ in range(8):
        x, y = torch.randn(6, 8), torch.randn(6, 4)
        opt_a.zero_grad()
        (pop_noop(x) - y).pow(2).mean().backward()
        opt_a.step()

        opt_b.zero_grad()
        (net_b(x) - y).pow(2).mean().backward()
        opt_b.step()

    for pa, pb in zip(net_a.parameters(), net_b.parameters()):
        assert torch.allclose(pa, pb, atol = 1e-6)

    # population mode exact parity against independent torch.optim.SGD instances
    nets = [make_mlp() for _ in range(4)]
    refs = [deepcopy(m) for m in nets]
    pop = Population(nets)
    opt = SGD(pop, lr = 0.05, momentum = momentum, weight_decay = 1e-2, nesterov = nesterov)
    ref_opts = [
        torch.optim.SGD(m.parameters(), lr = 0.05, momentum = momentum, weight_decay = 1e-2, nesterov = nesterov)
        for m in refs
    ]

    for _ in range(8):
        x_pop, y_pop = torch.randn(4, 6, 8), torch.randn(4, 6, 4)
        opt.zero_grad()
        (pop(x_pop, per_individual = True) - y_pop).pow(2).mean(dim = (1, 2)).sum().backward()
        opt.step()

        for i, (m, ro) in enumerate(zip(refs, ref_opts)):
            ro.zero_grad()
            (m(x_pop[i]) - y_pop[i]).pow(2).mean().backward()
            ro.step()

    for i, ref in enumerate(refs):
        for pa, pb in zip(pop[i].parameters(), ref.parameters()):
            assert torch.allclose(pa, pb, atol = 1e-6)


def test_sgd_extract_checkout_shared_and_functional_step():
    torch.manual_seed(1)
    nets = [make_mlp() for _ in range(3)]
    ref1 = deepcopy(nets[1])

    pop = Population(nets)
    opt = pop.sgd(lr = 0.05, momentum = 0.9, nesterov = True)
    ref_opt1 = torch.optim.SGD(ref1.parameters(), lr = 0.05, momentum = 0.9, nesterov = True)

    # Phase 1: train in population
    for _ in range(4):
        x_pop, y_pop = torch.randn(3, 6, 8), torch.randn(3, 6, 4)
        opt.zero_grad()
        (pop(x_pop, per_individual = True) - y_pop).pow(2).mean(dim = (1, 2)).sum().backward()
        opt.step()

        ref_opt1.zero_grad()
        (ref1(x_pop[1]) - y_pop[1]).pow(2).mean().backward()
        ref_opt1.step()

    # Phase 2: checkout member 1 -> native torch.optim.SGD solo steps -> auto writeback
    with pop.checkout(1) as (ind, ind_opt):
        assert isinstance(ind_opt, torch.optim.SGD)
        for _ in range(4):
            x, y = torch.randn(6, 8), torch.randn(6, 4)
            ind_opt.zero_grad()
            (ind(x) - y).pow(2).mean().backward()
            ind_opt.step()

            ref_opt1.zero_grad()
            (ref1(x) - y).pow(2).mean().backward()
            ref_opt1.step()

    # Phase 3: continue inside population -> exact trajectory continuity!
    for _ in range(4):
        x_pop, y_pop = torch.randn(3, 6, 8), torch.randn(3, 6, 4)
        opt.zero_grad()
        (pop(x_pop, per_individual = True) - y_pop).pow(2).mean(dim = (1, 2)).sum().backward()
        opt.step()

        ref_opt1.zero_grad()
        (ref1(x_pop[1]) - y_pop[1]).pow(2).mean().backward()
        ref_opt1.step()

    for pa, pb in zip(pop[1].parameters(), ref1.parameters()):
        assert torch.allclose(pa, pb, atol = 1e-5)

    # runtime share / unshare on SGD momentum buffer
    opt.share()
    assert pop.population_parameters.opt_slot_flat('momentum').ndim == 1
    opt.unshare()
    assert pop.population_parameters.opt_slot_flat('momentum').shape[0] == 3

    # differentiable functional_step through SGD with momentum
    meta = nn.Linear(4, 4)
    x = torch.randn(6, 8)
    cur_pp = pop.population_parameters
    for _ in range(2):
        loss = (pop(x, params = cur_pp) - meta(torch.ones(6, 4)).unsqueeze(0)).pow(2).mean()
        cur_pp = opt.functional_step(loss, cur_pp, create_graph = True)

    outer_loss = pop(x, params = cur_pp).pow(2).mean()
    outer_loss.backward()
    assert meta.weight.grad is not None and meta.weight.grad.abs().sum() > 0


# 2. Muon (Newton-Schulz on 2D+ matrices, AdamW on 1D params)

def test_muon_population_shared_subsets_lifecycle_and_functional_step():
    torch.manual_seed(2)
    sv = torch.linalg.svdvals(newton_schulz(torch.randn(16, 8)))
    assert sv.min() > 0.4 and sv.max() < 1.6

    pop = Population(make_mlp(), pop_size = 4, shared = ('0.bias',))
    opt = pop.muon(lr = [0.01, 0.02, 0.03, 0.04], weight_decay = 1e-2)
    pp = pop.population_parameters

    x = torch.randn(16, 8)
    y = torch.sin(x @ torch.randn(8, 4))

    init_loss = (pop(x) - y.unsqueeze(0)).pow(2).mean().item()
    for _ in range(15):
        opt.zero_grad()
        (pop(x) - y.unsqueeze(0)).pow(2).mean(dim = (1, 2)).sum().backward()
        opt.step()
    assert (pop(x) - y.unsqueeze(0)).pow(2).mean().item() < init_loss

    # subset step only updates selected members
    w_before = pop.params_dict['0.weight'].detach().clone()
    opt.zero_grad()
    (pop(x, members = (0, 2)) - y.unsqueeze(0)).pow(2).mean().backward()
    opt.step(members = (0, 2))
    assert not torch.allclose(pop.params_dict['0.weight'][0], w_before[0])
    assert torch.allclose(pop.params_dict['0.weight'][1], w_before[1])

    # delete & append resize Muon's matrix & vector slots in lockstep
    pop.delete(1)
    assert pop.pop_size == 3
    assert pp.opt_store('muon').slot_views('m')['0.weight'].shape == (3, 16, 8)
    assert pp.opt_store('muon').slot_views('m1')['2.bias'].shape == (3, 4)
    pop.append(0)
    assert pop.pop_size == 4

    # shared Muon slots + differentiable functional_step
    pop2 = Population(make_mlp(), pop_size = 3)
    opt2 = Muon(pop2, lr = 0.02, shared = True)
    assert pop2.population_parameters.opt_store('muon').slot_views('m')['0.weight'].shape == (16, 8)

    meta = nn.Linear(4, 4)
    cur_pp = pop2.population_parameters
    loss = (pop2(x, params = cur_pp) - meta(torch.ones(16, 4)).unsqueeze(0)).pow(2).mean()
    next_pp = opt2.functional_step(loss, cur_pp, create_graph = True)
    pop2(x, params = next_pp).pow(2).mean().backward()
    assert meta.weight.grad is not None and meta.weight.grad.abs().sum() > 0


# 3. Custom user-defined PopulationOptimizer (12-line Lion!)

def test_custom_population_optimizer_lion():
    class Lion(PopulationOptimizer):
        _valid_slots: ClassVar[frozenset[str]] = frozenset({'m'})

        def __init__(self, population, lr = 1e-3, betas = (0.9, 0.99), weight_decay = 0., *, shared = False, name = 'lion'):
            self.beta1, self.beta2 = betas
            super().__init__(
                population,
                shared = shared,
                name = name,
                hp_specs = {'lr': lr, 'weight_decay': weight_decay},
                defaults = dict(lr = lr, weight_decay = weight_decay),
            )

        def _init_store_slots(self, store):
            store.alloc_slot('m', shared = 'm' in self._shared_slots)

        def _update(self, p, g, slots, step, hps, *, spec = None, is_pop = False, stop_grad_denom = False):
            lr, wd = hps['lr'], hps['weight_decay']
            m = slots['m']
            if wd:
                p = p * (1. - lr * wd)
            u = (m * self.beta1 + g * (1. - self.beta1)).sign()
            return p - lr * u, {'m': m * self.beta2 + g * (1. - self.beta2)}

        def _spawn_sub_optimizer(self, target_pop, indices):
            return Lion(target_pop, lr = self._resolve_hp_dict()['lr'], betas = (self.beta1, self.beta2), shared = self._shared_slots, name = self.name)

    torch.manual_seed(3)
    pop = Population(make_mlp(), pop_size = 4)
    opt = Lion(pop, lr = 1e-2, shared = False)

    x = torch.randn(16, 8)
    y = torch.randn(16, 4)

    init_loss = (pop(x) - y.unsqueeze(0)).pow(2).mean().item()
    for _ in range(20):
        opt.zero_grad()
        (pop(x) - y.unsqueeze(0)).pow(2).mean(dim = (1, 2)).sum().backward()
        opt.step()

    assert (pop(x) - y.unsqueeze(0)).pow(2).mean().item() < init_loss

    # Lion automatically gets share(), unshare(), delete(), append(), and functional_step()!
    opt.share()
    assert pop.population_parameters.opt_slot_flat('m').ndim == 1
    opt.unshare()
    pop.delete(0)
    assert pop.population_parameters.opt_slot_flat('m').shape[0] == 3
    next_pp = opt.functional_step(pop(x).pow(2).mean(), create_graph = True)
    assert next_pp.flat_params.shape[0] == 3
