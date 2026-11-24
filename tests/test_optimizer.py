from copy import deepcopy

import pytest
import torch
from torch import nn
from torch.nn import Module
from torch.optim import Optimizer

from torch_population import ParamSlot, Population, PopulationOptimizer

param = pytest.mark.parametrize

# fixtures

def mlp():
    return nn.Sequential(
        nn.Linear(8, 16),
        nn.SiLU(),
        nn.Linear(16, 4),
    )

def step_mse(mod, opt, x, y = None, *, per_individual = False, **step_kw):
    opt.zero_grad()
    is_pop = isinstance(mod, Population) and mod.is_population
    out = mod(x, per_individual = per_individual) if (is_pop and per_individual) else mod(x)
    if y is None:
        diff = out
    else:
        target = y if (per_individual or out.ndim == y.ndim) else y.unsqueeze(0)
        diff = out - target
    loss = diff.pow(2).mean(dim = (1, 2)).sum() if is_pop else diff.pow(2).mean()
    loss.backward()
    opt.step(**step_kw)

class MatrixOrthOpt(Optimizer):
    """Shape-sensitive 2D+ optimizer (auto-selects per-individual execution)."""

    def __init__(self, params, lr = 1e-2, decay = 0.9):
        super().__init__(params, dict(lr = lr, decay = decay))

    @torch.no_grad()
    def step(self, closure = None):
        for group in self.param_groups:
            lr, decay = group['lr'], group['decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                buf = self.state[p].setdefault('vel', torch.zeros_like(p))
                buf.mul_(decay).add_(p.grad)
                u = buf
                if p.ndim >= 2:
                    u = u / u.norm().clamp(min = 1e-7)
                    u = 1.5 * u - 0.5 * (u @ u.mT) @ u
                p.add_(u, alpha = -lr)

# tests

@param('opt_cls,kw,expect_batch', [
    (torch.optim.Adam, dict(lr = 1e-2, betas = (0.9, 0.99), weight_decay = 1e-2), True),
    (torch.optim.AdamW, dict(lr = 1e-2, betas = (0.9, 0.99), weight_decay = 2e-2), True),
    (torch.optim.SGD, dict(lr = 5e-2, momentum = 0., weight_decay = 1e-2), True),
    (torch.optim.SGD, dict(lr = 5e-2, momentum = 0.9, nesterov = True, weight_decay = 1e-2), True),
    (torch.optim.RMSprop, dict(lr = 1e-2, momentum = 0.5), True),
    (torch.optim.Adagrad, dict(lr = 5e-2), True),
    (torch.optim.Adafactor, dict(lr = 1e-2), False),
    (MatrixOrthOpt, dict(lr = 2e-2, decay = 0.9), False),
])
def test_optimizer_parity(opt_cls, kw, expect_batch):
    torch.manual_seed(0)

    # noop parity
    net_a = mlp()
    net_b = deepcopy(net_a)
    pop_noop = Population(net_a)
    opt_a = pop_noop.optimizer(opt_cls, **kw)
    opt_b = opt_cls(net_b.parameters(), **kw)

    for _ in range(6):
        x, y = torch.randn(6, 8), torch.randn(6, 4)
        step_mse(pop_noop, opt_a, x, y)
        step_mse(net_b, opt_b, x, y)
    assert all(torch.allclose(a, b, atol = 1e-5) for a, b in zip(pop_noop.parameters(), net_b.parameters()))

    # population parity
    nets = [mlp() for _ in range(4)]
    refs = [deepcopy(m) for m in nets]
    pop = Population(nets)
    opt = PopulationOptimizer(pop, opt_cls, **kw)
    ref_opts = [opt_cls(m.parameters(), **kw) for m in refs]

    assert opt._can_batch is expect_batch
    assert not opt.is_shared

    for _ in range(6):
        xp, yp = torch.randn(4, 6, 8), torch.randn(4, 6, 4)
        step_mse(pop, opt, xp, yp, per_individual = True)
        for i, (m, ro) in enumerate(zip(refs, ref_opts)):
            step_mse(m, ro, xp[i], yp[i])

    for i, ref in enumerate(refs):
        assert all(torch.allclose(a, b, atol = 1e-5) for a, b in zip(pop[i].parameters(), ref.parameters()))

def test_colocated_storage_state_dict_and_dtype():
    torch.manual_seed(2)
    pop = Population(mlp(), pop_size = 3)
    n_params = sum(p.numel() for p in pop.parameters())
    opt = pop.optimizer(torch.optim.RMSprop, lr = 1e-2, momentum = 0.5)

    assert sum(p.numel() for p in pop.parameters()) == n_params
    assert pop.opt is opt

    slot = pop.slot('0.weight')
    assert isinstance(slot, ParamSlot)
    assert slot.param.shape == (3, 16, 8)
    assert slot.state['square_avg'].shape == (3, 16, 8)
    assert slot.state['momentum_buffer'].shape == (3, 16, 8)
    assert slot.state['step'].shape == (3,)

    x = torch.randn(5, 8)
    for _ in range(3):
        step_mse(pop, opt, x)
    assert torch.allclose(pop.slot('0.weight').state['step'], torch.tensor([3., 3., 3.]))

    # state_dict round-trip restores weights + colocated opt state
    pop2 = Population(mlp(), pop_size = 3)
    opt2 = pop2.optimizer(torch.optim.RMSprop, lr = 1e-2, momentum = 0.5)
    pop2.load_state_dict(pop.state_dict())

    st1 = opt.named_state(pop.population_parameters)
    st2 = opt2.named_state(pop2.population_parameters)
    for name in pop.params_dict:
        assert torch.equal(pop2.params_dict[name], pop.params_dict[name])
        assert all(torch.equal(st2[name][k], st1[name][k]) for k in st1[name])

    # dtype conversion
    pop.double()
    slot64 = pop.slot('0.weight')
    assert slot64.param.dtype == torch.float64
    assert slot64.state['square_avg'].dtype == torch.float64

    step_mse(pop, opt, x.double())
    assert torch.allclose(pop.slot('0.weight').state['step'], torch.tensor([4., 4., 4.], dtype = torch.float64))

def test_sharing_modes_and_runtime_toggle():
    torch.manual_seed(3)
    x = torch.randn(32, 8)
    y = torch.sin(x @ torch.randn(8, 4))

    # fully shared (1/P memory)
    pop_s = Population(mlp(), pop_size = 4, init = 'clone', noise_std = 0.02)
    opt_s = pop_s.optimizer(torch.optim.RMSprop, lr = 1e-2, shared = True)
    st_s = opt_s.named_state(pop_s.population_parameters)['0.weight']
    assert opt_s.is_shared
    assert st_s['square_avg'].shape == (16, 8)
    assert st_s['step'].shape == ()

    loss0 = (pop_s(x) - y.unsqueeze(0)).pow(2).mean().item()
    for _ in range(100):
        step_mse(pop_s, opt_s, x, y)
    assert (pop_s(x) - y.unsqueeze(0)).pow(2).mean().item() < loss0 * 0.2

    # granular sharing by state key
    pop_g = Population(mlp(), pop_size = 4)
    opt_g = pop_g.optimizer(torch.optim.RMSprop, lr = 1e-2, momentum = 0.5, shared = ('square_avg',))
    st_g = opt_g.named_state(pop_g.population_parameters)['0.weight']
    assert not opt_g.is_shared
    assert opt_g.shared_keys == frozenset({'square_avg'})
    assert st_g['square_avg'].shape == (16, 8)
    assert st_g['momentum_buffer'].shape == (4, 16, 8)
    assert st_g['step'].shape == (4,)

    loss_g0 = (pop_g(x) - y.unsqueeze(0)).pow(2).mean().item()
    for _ in range(80):
        step_mse(pop_g, opt_g, x, y)
    assert (pop_g(x) - y.unsqueeze(0)).pow(2).mean().item() < loss_g0 * 0.25

    # runtime separate -> shared -> separate
    pop_rt = Population(mlp(), pop_size = 3)
    opt_rt = pop_rt.optimizer(torch.optim.RMSprop, lr = 1e-2, shared = False)
    step_mse(pop_rt, opt_rt, x)

    sq0 = pop_rt.slot('0.weight').state['square_avg'].clone()
    opt_rt.share()
    assert opt_rt.is_shared
    assert torch.allclose(pop_rt.slot('0.weight').state['square_avg'], sq0.mean(0))
    assert pop_rt.slot('0.weight').state['step'].shape == ()

    opt_rt.unshare()
    sq_u = pop_rt.slot('0.weight').state['square_avg']
    assert not opt_rt.is_shared
    assert sq_u.shape == (3, 16, 8)
    assert torch.allclose(sq_u[0], sq_u[2])
    assert pop_rt.slot('0.weight').state['step'].shape == (3,)

@param('opt_cls,kw', [
    (torch.optim.AdamW, dict(lr = 1e-2, betas = (0.9, 0.999), weight_decay = 1e-2)),
    (torch.optim.SGD, dict(lr = 5e-2, momentum = 0.9, nesterov = True)),
    (torch.optim.RMSprop, dict(lr = 1e-2, momentum = 0.5)),
    (MatrixOrthOpt, dict(lr = 2e-2, decay = 0.9)),
])
def test_checkout_and_writeback_continuity(opt_cls, kw):
    torch.manual_seed(4)
    nets = [mlp() for _ in range(4)]
    ref = deepcopy(nets[2])
    pop = Population(nets)
    opt = pop.optimizer(opt_cls, **kw)
    ref_opt = opt_cls(ref.parameters(), **kw)

    for _ in range(4):
        xp, yp = torch.randn(4, 6, 8), torch.randn(4, 6, 4)
        step_mse(pop, opt, xp, yp, per_individual = True)
        step_mse(ref, ref_opt, xp[2], yp[2])

    others = pop.params_dict['0.weight'][[0, 1, 3]].detach().clone()

    with pop.checkout(2) as (ind, ind_opt):
        assert isinstance(ind, nn.Sequential)
        assert isinstance(ind_opt, opt_cls)
        for _ in range(4):
            xs, ys = torch.randn(6, 8), torch.randn(6, 4)
            step_mse(ind, ind_opt, xs, ys)
            step_mse(ref, ref_opt, xs, ys)

    assert torch.equal(pop.params_dict['0.weight'][[0, 1, 3]], others)

    for _ in range(4):
        xp, yp = torch.randn(4, 6, 8), torch.randn(4, 6, 4)
        step_mse(pop, opt, xp, yp, per_individual = True)
        step_mse(ref, ref_opt, xp[2], yp[2])

    assert all(torch.allclose(a, b, atol = 1e-5) for a, b in zip(pop[2].parameters(), ref.parameters()))

def test_insert_unstepped_optimizer_zeroes_slot():
    pop = Population(mlp(), pop_size = 3)
    opt = pop.optimizer(torch.optim.RMSprop, lr = 1e-2)
    step_mse(pop, opt, torch.randn(5, 8))
    assert torch.allclose(pop.slot('0.weight').state['step'], torch.tensor([1., 1., 1.]))

    fresh = mlp()
    pop[1] = (fresh, torch.optim.RMSprop(fresh.parameters(), lr = 1e-2))
    assert torch.allclose(pop.slot('0.weight').state['step'], torch.tensor([1., 0., 1.]))
    assert torch.equal(pop.slot('0.weight').state['square_avg'][1], torch.zeros(16, 8))

def test_populate_and_slice_carry_opt():
    torch.manual_seed(6)
    x = torch.randn(5, 8)

    # noop -> step -> populate(4)
    pop = Population(mlp())
    opt = pop.optimizer(torch.optim.RMSprop, lr = 1e-2)
    for _ in range(3):
        step_mse(pop, opt, x)

    sq0 = pop.slot('0.weight').state['square_avg'].clone()
    pop.populate(4, init = 'reinit')
    assert torch.equal(pop.slot('0.weight').state['square_avg'][0], sq0)
    assert torch.equal(pop.slot('0.weight').state['square_avg'][1], torch.zeros_like(sq0))
    assert torch.allclose(pop.slot('0.weight').state['step'], torch.tensor([3., 0., 0., 0.]))

    step_mse(pop, opt, x)
    assert torch.allclose(pop.slot('0.weight').state['step'], torch.tensor([4., 1., 1., 1.]))

    # slice resample + slice writeback carry opt state & per-individual HPs
    pop4 = Population(mlp(), pop_size = 4)
    opt4 = pop4.optimizer(torch.optim.RMSprop, lr = [1e-3, 5e-3, 1e-2, 2e-2])
    step_mse(pop4, opt4, x)

    sub = pop4[[3, 0, 3]]
    assert sub.opt is not None
    assert torch.equal(sub.slot('0.weight').state['square_avg'][0], pop4.slot('0.weight').state['square_avg'][3])
    assert torch.equal(sub.slot('0.weight').state['square_avg'][1], pop4.slot('0.weight').state['square_avg'][0])
    assert torch.allclose(sub.opt.param_groups[0]['lr'], torch.tensor([2e-2, 1e-3, 2e-2]))

    for _ in range(2):
        step_mse(sub, sub.opt, x)
    pop4[0:2] = sub[0:2]
    assert torch.allclose(pop4.slot('0.weight').state['step'], torch.tensor([3., 3., 1., 1.]))
    assert torch.equal(pop4.slot('0.weight').state['square_avg'][0], sub.slot('0.weight').state['square_avg'][0])

def test_subset_step_reset_hp_scheduler_and_external_grads():
    torch.manual_seed(8)
    pop = Population(mlp(), pop_size = 4, init = 'clone')
    opt = pop.optimizer(torch.optim.RMSprop, lr = [1e-4, 1e-3, 1e-2, 5e-2], alpha = [0.9, 0.95, 0.98, 0.99])

    opt.zero_grad()
    pop.view([0, 2])(torch.randn(6, 8)).pow(2).mean().backward()
    opt.step(members = [0, 2])
    assert torch.allclose(pop.slot('0.weight').state['step'], torch.tensor([1., 0., 1., 0.]))
    assert torch.equal(pop.slot('0.weight').state['square_avg'][1], torch.zeros(16, 8))

    _, ind_opt3 = pop.extract(3)
    assert ind_opt3.param_groups[0]['lr'] == pytest.approx(5e-2)
    assert ind_opt3.param_groups[0]['alpha'] == pytest.approx(0.99)

    opt.reset(torch.tensor([True, False, False, False]))
    assert torch.allclose(pop.slot('0.weight').state['step'], torch.tensor([0., 0., 1., 0.]))
    assert torch.equal(pop.slot('0.weight').state['square_avg'][0], torch.zeros(16, 8))
    assert pop.slot('0.weight').state['square_avg'][2].abs().sum() > 0

    # external grads + StepLR (including pop_size == 2 with tuple betas)
    pop2 = Population(mlp(), pop_size = 2)
    opt2 = pop2.optimizer(torch.optim.AdamW, lr = 1e-2, betas = (0.9, 0.999))
    sched = torch.optim.lr_scheduler.StepLR(opt2, step_size = 1, gamma = 0.5)
    opt2.step(grads = {k: torch.ones_like(p) for k, p in pop2.params_dict.items()})
    sched.step()
    assert opt2.param_groups[0]['lr'] == pytest.approx(5e-3)
    assert torch.allclose(pop2.slot('0.weight').state['step'], torch.tensor([1., 1.]))

def test_noop_extract_and_frozen_or_unused_params():
    torch.manual_seed(9)
    x = torch.randn(5, 8)

    net = mlp()
    pop = Population(net)
    opt = pop.optimizer(torch.optim.RMSprop, lr = 1e-2)
    for _ in range(2):
        step_mse(pop, opt, x)

    ind, ind_opt = pop.extract()
    assert ind is net
    for _ in range(3):
        step_mse(ind, ind_opt, x)
    pop[0] = (ind, ind_opt)
    assert int(pop.slot('0.weight').state['step'].item()) == 5

    class TwoHead(Module):
        def __init__(self):
            super().__init__()
            self.head_a = nn.Linear(8, 4)
            self.head_b = nn.Linear(8, 4)
            self.head_a.bias.requires_grad_(False)

        def forward(self, x, use_b = False):
            return self.head_b(x) if use_b else self.head_a(x)

    pop_f = Population(TwoHead(), pop_size = 3)
    opt_f = pop_f.optimizer(torch.optim.RMSprop, lr = 1e-2)
    assert not pop_f.slot('head_a.bias').state

    opt_f.zero_grad()
    pop_f(x, use_b = False).pow(2).mean().backward()
    opt_f.step()
    assert torch.allclose(pop_f.slot('head_a.weight').state['step'], torch.tensor([1., 1., 1.]))
    assert torch.allclose(pop_f.slot('head_b.weight').state['step'], torch.tensor([0., 0., 0.]))

    ind_f, ind_opt_f = pop_f.extract(1)
    assert not ind_f.head_a.bias.requires_grad
    ind_opt_f.zero_grad()
    ind_f(x, use_b = True).pow(2).mean().backward()
    ind_opt_f.step()
    pop_f[1] = (ind_f, ind_opt_f)
    assert torch.allclose(pop_f.slot('head_b.weight').state['step'], torch.tensor([0., 1., 0.]))

@param('opt_cls,kw', [
    (torch.optim.Adam, dict(lr = 1e-2)),
    (torch.optim.AdamW, dict(lr = 1e-2, weight_decay = 1e-2)),
    (torch.optim.SGD, dict(lr = 5e-2, momentum = 0.9, nesterov = True)),
    (torch.optim.RMSprop, dict(lr = 1e-2)),
    (torch.optim.Adagrad, dict(lr = 5e-2)),
])
def test_differentiable_functional_step(opt_cls, kw):
    torch.manual_seed(10)
    pop = Population(mlp(), pop_size = 3)
    opt = pop.optimizer(opt_cls, **kw)
    meta = nn.Linear(4, 4)
    x = torch.randn(6, 8)

    cur_pp = pop.population_parameters
    for _ in range(2):
        loss = (pop(x, params = cur_pp) - meta(torch.ones(6, 4)).unsqueeze(0)).pow(2).mean()
        cur_pp = opt.functional_step(loss, cur_pp, create_graph = True)

    pop(x, params = cur_pp).pow(2).mean().backward()
    assert meta.weight.grad is not None
    assert meta.weight.grad.abs().sum() > 0

@param('opt_cls,kw,expect_hetero', [
    (torch.optim.Adam, dict(lr = 1e-2, betas = (0.9, 0.99)), True),
    (torch.optim.AdamW, dict(lr = 1e-2, weight_decay = 1e-2), True),
    (torch.optim.RMSprop, dict(lr = 1e-2, momentum = 0.5), True),
    (torch.optim.SGD, dict(lr = 5e-2, momentum = 0.9), True),
    (torch.optim.SGD, dict(lr = 5e-2, momentum = 0.9, dampening = 0.5), False),
    (torch.optim.Adagrad, dict(lr = 5e-2), False),
])
def test_post_reset_heterogeneous_step_and_functional_parity(opt_cls, kw, expect_hetero):
    torch.manual_seed(12)
    nets = [mlp() for _ in range(4)]
    refs = [deepcopy(m) for m in nets]
    pop = Population(nets)
    opt = pop.optimizer(opt_cls, **kw)
    ref_opts = [opt_cls(m.parameters(), **kw) for m in refs]

    assert opt._can_batch_hetero is expect_hetero

    # warm up all 4 members for 3 steps
    for _ in range(3):
        xp, yp = torch.randn(4, 6, 8), torch.randn(4, 6, 4)
        step_mse(pop, opt, xp, yp, per_individual = True)
        for i, (m, ro) in enumerate(zip(refs, ref_opts)):
            step_mse(m, ro, xp[i], yp[i])

    # reset members 0 and 2 (creates heterogeneous step counts & init flags)
    mask = torch.tensor([True, False, True, False])
    pop.reset_(mask)
    for i in (0, 2):
        refs[i] = pop[i]
        ref_opts[i] = opt_cls(refs[i].parameters(), **kw)

    # verify both inplace step() and functional_step(create_graph=True) match ground truth
    pp_fn = pop.population_parameters.clone()
    for _ in range(4):
        xp, yp = torch.randn(4, 6, 8), torch.randn(4, 6, 4)
        step_mse(pop, opt, xp, yp, per_individual = True)

        fn_loss = (pop(xp, params = pp_fn, per_individual = True) - yp).pow(2).mean(dim = (1, 2)).sum()
        pp_fn = opt.functional_step(fn_loss, pp_fn, create_graph = True)

        for i, (m, ro) in enumerate(zip(refs, ref_opts)):
            step_mse(m, ro, xp[i], yp[i])

    for i, ref in enumerate(refs):
        for name, p_ref in ref.named_parameters():
            assert torch.allclose(pop.params_dict[name][i], p_ref, atol = 1e-5)
            assert torch.allclose(pp_fn.named_parameters_dict()[name][i], p_ref, atol = 1e-5)

    # meta-gradients flow through post-reset hetero functional_step
    meta = nn.Linear(4, 4)
    cur_pp = pop.population_parameters.clone()
    x = torch.randn(4, 6, 8)
    for _ in range(2):
        loss = (pop(x, params = cur_pp, per_individual = True) - meta(torch.ones(6, 4)).unsqueeze(0)).pow(2).mean()
        cur_pp = opt.functional_step(loss, cur_pp, create_graph = True)
    pop(x, params = cur_pp, per_individual = True).pow(2).mean().backward()
    assert meta.weight.grad is not None and meta.weight.grad.abs().sum() > 0

def test_functional_step_with_shared_layers():
    torch.manual_seed(13)
    pop = Population(mlp(), pop_size = 3, shared = ('0.bias',))
    opt = pop.optimizer(torch.optim.Adam, lr = 1e-2)
    meta = nn.Linear(4, 4)
    x = torch.randn(6, 8)

    pp0 = pop.population_parameters
    b0 = pp0.flat_shared_params.detach().clone()
    w0 = pp0.flat_params.detach().clone()

    cur_pp = pp0
    for _ in range(2):
        loss = (pop(x, params = cur_pp) - meta(torch.ones(6, 4)).unsqueeze(0)).pow(2).mean()
        cur_pp = opt.functional_step(loss, cur_pp, create_graph = True)

    assert not torch.allclose(cur_pp.flat_params, w0)
    assert not torch.allclose(cur_pp.flat_shared_params, b0)

    pop(x, params = cur_pp).pow(2).mean().backward()
    assert meta.weight.grad is not None and meta.weight.grad.abs().sum() > 0

@param('detach_preconditioner', [True, False])
@param('opt_cls,kw', [
    (torch.optim.Adam, dict(lr = 1e-2)),
    (torch.optim.AdamW, dict(lr = 1e-2, weight_decay = 1e-2)),
    (torch.optim.RMSprop, dict(lr = 1e-2)),
    (torch.optim.Adagrad, dict(lr = 5e-2)),
])
def test_functional_step_zero_grad_entries_no_nan(opt_cls, kw, detach_preconditioner):
    torch.manual_seed(14)
    pop = Population(nn.Linear(8, 4), pop_size = 3)
    opt = pop.optimizer(opt_cls, **kw)
    meta = nn.Linear(4, 4)

    # zero out half of the input dims so half of weight.grad is identically 0 on step 1
    x = torch.randn(6, 8)
    x[:, 4:] = 0.

    cur_pp = pop.population_parameters
    for _ in range(2):
        loss = (pop(x, params = cur_pp) - meta(torch.ones(6, 4)).unsqueeze(0)).pow(2).mean()
        cur_pp = opt.functional_step(loss, cur_pp, create_graph = True, detach_preconditioner = detach_preconditioner)

    pop(torch.randn(6, 8), params = cur_pp).pow(2).mean().backward()
    assert meta.weight.grad is not None
    assert torch.isfinite(meta.weight.grad).all()
    assert meta.weight.grad.abs().sum() > 0

def test_shared_step_reset_preserves_step_and_factory_or_anchor_reset():
    class CustomGain(Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(8, 4)
            self.gain = nn.Parameter(torch.zeros(4))

        def forward(self, x):
            return self.lin(x) * (1. + self.gain)

    torch.manual_seed(15)
    pop = Population(lambda: CustomGain(), pop_size = 4)
    opt = pop.optimizer(torch.optim.Adam, lr = 1e-2, shared = ('step',))
    pp = pop.population_parameters

    x = torch.randn(4, 6, 8)
    for _ in range(3):
        step_mse(pop, opt, x, per_individual = True)

    assert pp.opt_aux('step', 'lin.weight').item() == 3.
    with torch.no_grad():
        pp.named_parameters_dict()['gain'][0].fill_(5.0)

    # factory reset zeroes custom parameter without reset_parameters() and preserves shared step
    mask = torch.tensor([True, False, False, False])
    pop.reset_(mask)
    assert torch.equal(pp.named_parameters_dict()['gain'][0], torch.zeros(4))
    assert pp.opt_slot_flat('exp_avg')[0].norm() == 0
    assert pp.opt_slot_flat('exp_avg')[1].norm() > 0
    assert pp.opt_aux('step', 'lin.weight').item() == 3.

    # next step increments shared step cleanly to 4 (not corrupted backwards)
    step_mse(pop, opt, x, per_individual = True)
    assert pp.opt_aux('step', 'lin.weight').item() == 4.

    # anchor-module reset copies exact weights into masked slots
    anchor = CustomGain()
    pop.reset_(torch.tensor([False, True, False, True]), init = anchor)
    assert torch.allclose(pp.named_parameters_dict()['lin.weight'][1], anchor.lin.weight)
    assert torch.allclose(pp.named_parameters_dict()['lin.weight'][3], anchor.lin.weight)

    # population-reference reset copies vectorized from pristine population and preserves shared step
    ref_pop = Population(lambda: CustomGain(), pop_size = 4)
    pop.reset_(torch.tensor([True, False, False, True]), init = ref_pop)
    assert torch.allclose(pp.named_parameters_dict()['lin.weight'][0], ref_pop.params_dict['lin.weight'][0])
    assert torch.allclose(pp.named_parameters_dict()['lin.weight'][3], ref_pop.params_dict['lin.weight'][3])
    assert pp.opt_aux('step', 'lin.weight').item() == 4.

    # full reset_() without selector resets shared step back to 0
    pop.reset_()
    assert pp.opt_aux('step', 'lin.weight').item() == 0.

    # deferred Population(factory).populate(P) also uses factory for custom layers
    pop_deferred = Population(lambda: CustomGain())
    with torch.no_grad():
        pop_deferred.unwrap().gain.fill_(9.0)
    pop_deferred.populate(3)
    assert torch.equal(pop_deferred.params_dict['gain'][0], torch.full((4,), 9.0))
    assert torch.equal(pop_deferred.params_dict['gain'][1], torch.zeros(4))
    assert torch.equal(pop_deferred.params_dict['gain'][2], torch.zeros(4))

def test_share_reductions_and_post_reset_share():
    torch.manual_seed(16)
    pop = Population(mlp(), pop_size = 4)
    opt = pop.optimizer(torch.optim.RMSprop, lr = 1e-2, shared = False)
    x = torch.randn(4, 6, 8)
    for _ in range(3):
        step_mse(pop, opt, x, per_individual = True)

    sq = pop.slot('0.weight').state['square_avg'].clone()

    # per-key reduction: max on square_avg & step
    opt.share(reduce = {'square_avg': 'max', 'step': 'max'})
    assert torch.allclose(pop.slot('0.weight').state['square_avg'], sq.amax(dim = 0))
    assert pop.slot('0.weight').state['step'].item() == 3.

    # unshare and share by member index, custom callable, and zero
    opt.unshare()
    with torch.no_grad():
        pop.slot('0.weight').state['square_avg'].copy_(sq)
    opt.share(reduce = 2)
    assert torch.allclose(pop.slot('0.weight').state['square_avg'], sq[2])

    opt.unshare()
    with torch.no_grad():
        pop.slot('0.weight').state['square_avg'].copy_(sq)
    opt.share(reduce = lambda t: t.sum(dim = 0))
    assert torch.allclose(pop.slot('0.weight').state['square_avg'], sq.sum(dim = 0))

    opt.unshare()
    opt.share(reduce = 'zero')
    assert pop.slot('0.weight').state['square_avg'].norm() == 0

    # sharing after a partial reset synchronizes _init and does not corrupt shared step on next step
    opt.unshare()
    for _ in range(2):
        step_mse(pop, opt, x, per_individual = True)
    pop.reset_(torch.tensor([True, False, False, False]))
    assert torch.allclose(pop.slot('0.weight').state['step'], torch.tensor([0., 2., 2., 2.]))
    opt.share(reduce = 'max')
    assert pop.slot('0.weight').state['step'].item() == 2.
    step_mse(pop, opt, x, per_individual = True)
    assert pop.slot('0.weight').state['step'].item() == 3.

    # shared_reduce='max' governs ongoing shared steps too
    pop_m = Population(mlp(), pop_size = 3)
    opt_m = pop_m.optimizer(torch.optim.RMSprop, lr = 1e-2, shared = True, shared_reduce = 'max')
    step_mse(pop_m, opt_m, torch.randn(3, 6, 8), per_individual = True)
    assert pop_m.slot('0.weight').state['square_avg'].shape == (16, 8)

    # shared parameter ('0.bias') does NOT prevent masked reset from clearing _init for per-member weights
    pop_sp = Population(mlp(), pop_size = 3, shared = ('0.bias',))
    opt_sp = pop_sp.optimizer(torch.optim.Adam, lr = 1e-2, shared = False)
    step_mse(pop_sp, opt_sp, torch.randn(3, 6, 8), per_individual = True)
    pop_sp.reset_(torch.tensor([True, False, False]))
    store = pop_sp.population_parameters.opt_stores[opt_sp.name]
    assert not store.is_init(pop_sp.population_parameters._schema.by_name['0.weight'], 0)
    assert store.is_init(pop_sp.population_parameters._schema.by_name['0.weight'], 1)
    assert store.is_init(pop_sp.population_parameters._schema.by_name['0.bias'])

def test_autograd_multi_loss_pcgrad_and_functional_step_grads():
    torch.manual_seed(17)
    pop = Population(mlp(), pop_size = 3)
    opt = pop.optimizer(torch.optim.Adam, lr = 1e-2)
    meta = nn.Linear(4, 4)
    pp = pop.population_parameters

    x = torch.randn(3, 6, 8)
    out = pop(x, params = pp, per_individual = True)
    loss_a = (out - meta(torch.ones(6, 4)).unsqueeze(0)).pow(2).mean()
    loss_b = (out + 0.5).pow(2).mean()

    g_a = pp.autograd(loss_a, create_graph = True, retain_graph = True)
    g_b = pp.autograd(loss_b, create_graph = True)

    # per-member PCGrad projection (differentiable!)
    combined = {}
    for k in g_a:
        ga, gb = g_a[k], g_b[k]
        dot = (ga * gb).flatten(1).sum(dim = -1)
        norm_sq = (ga * ga).flatten(1).sum(dim = -1).clamp_min(1e-12)
        proj = (dot.clamp_max(0.) / norm_sq).view(-1, *(1,) * (ga.ndim - 1)) * ga
        combined[k] = ga + (gb - proj)

    next_pp = opt.functional_step(grads = combined, params = pp, create_graph = True)
    pop(x, params = next_pp, per_individual = True).pow(2).mean().backward()
    assert meta.weight.grad is not None and meta.weight.grad.abs().sum() > 0

    # scoped partial grads only update the targeted parameters
    pp0 = pop.population_parameters.clone()
    w0_before = pp0.named_parameters_dict()['0.weight'].detach().clone()
    w2_before = pp0.named_parameters_dict()['2.weight'].detach().clone()
    scoped_pp = opt.functional_step(grads = {'2.weight': torch.ones_like(w2_before)}, params = pp0, create_graph = False)
    assert torch.equal(scoped_pp.named_parameters_dict()['0.weight'], w0_before)
    assert not torch.allclose(scoped_pp.named_parameters_dict()['2.weight'], w2_before)
    assert scoped_pp.opt_aux('step', '0.weight').sum().item() == 0.
    assert (scoped_pp.opt_aux('step', '2.weight') == 1.).all()

def test_functional_step_subset_members():
    torch.manual_seed(18)
    pop = Population(mlp(), pop_size = 4)
    opt = pop.optimizer(torch.optim.Adam, lr = 1e-2)
    meta = nn.Linear(4, 4)
    pp0 = pop.population_parameters
    w_before = pp0.flat_params.detach().clone()

    x = torch.randn(4, 6, 8)
    out = pop(x, params = pp0, per_individual = True)
    loss = (out - meta(torch.ones(6, 4)).unsqueeze(0)).pow(2).mean()
    pp1 = opt.functional_step(loss, pp0, members = (0, 2), create_graph = True)

    assert not torch.allclose(pp1.flat_params[0], w_before[0])
    assert not torch.allclose(pp1.flat_params[2], w_before[2])
    assert torch.equal(pp1.flat_params[1], w_before[1])
    assert torch.equal(pp1.flat_params[3], w_before[3])
    assert torch.allclose(pp1.opt_aux('step', '0.weight'), torch.tensor([1., 0., 1., 0.]))
    assert pp1.opt_slot_flat('exp_avg')[0].norm() > 0
    assert pp1.opt_slot_flat('exp_avg')[1].norm() == 0

    # meta-gradients flow through updated subset members
    pop(x, params = pp1, per_individual = True).pow(2).mean().backward()
    assert meta.weight.grad is not None and meta.weight.grad.abs().sum() > 0


class PyScalarStepOpt(torch.optim.Optimizer):
    '''Minimal optimizer storing Python int/float state (like AdoptAtan2's steps) and lazy param_init.'''
    def __init__(self, params, lr = 1e-2, beta = 0.9, regen = 1e-3):
        super().__init__(params, dict(lr = lr, beta = beta, regen = regen))

    @torch.no_grad()
    def step(self, closure = None):
        for group in self.param_groups:
            lr, beta, regen = group['lr'], group['beta'], group['regen']
            for p in group['params']:
                if p.grad is None:
                    continue
                st = self.state[p]
                if len(st) == 0:
                    st['steps'] = 0
                    st['scale'] = 1.0
                    st['exp_avg'] = torch.zeros_like(p)
                    st['param_init'] = p.data.clone()
                if st['steps'] == 0:
                    st['exp_avg'].copy_(p.grad)
                    st['steps'] += 1
                    continue
                st['exp_avg'].mul_(beta).add_(p.grad, alpha = 1. - beta)
                p.add_(st['exp_avg'], alpha = -lr * st['scale'])
                p.lerp_(st['param_init'], lr * regen)
                st['steps'] += 1
                st['scale'] = 0.99 * st['scale']


class OutOfPlaceStateOpt(torch.optim.Optimizer):
    '''Optimizer that replaces state tensors out-of-place instead of mutating in-place.'''
    def __init__(self, params, lr = 1e-2, beta = 0.9):
        super().__init__(params, dict(lr = lr, beta = beta))

    @torch.no_grad()
    def step(self, closure = None):
        for group in self.param_groups:
            lr, beta = group['lr'], group['beta']
            for p in group['params']:
                if p.grad is None:
                    continue
                st = self.state[p]
                buf = st.get('buf', torch.zeros_like(p))
                buf = beta * buf + (1. - beta) * p.grad
                st['buf'] = buf
                p.add_(buf, alpha = -lr)


def test_python_scalar_state_and_noop_cache_lifecycle():
    torch.manual_seed(19)
    base = mlp()

    # 1. noop parity (with flat autograd + LRScheduler + state_dict reload + dtype sync + reset)
    m_ref, m_pop = deepcopy(base), deepcopy(base)
    o_ref = PyScalarStepOpt(m_ref.parameters(), lr = 2e-2)
    pop_noop = Population(m_pop)
    o_pop = pop_noop.optimizer(PyScalarStepOpt, lr = 2e-2)
    sched_ref = torch.optim.lr_scheduler.StepLR(o_ref, step_size = 2, gamma = 0.5)
    sched_pop = torch.optim.lr_scheduler.StepLR(o_pop, step_size = 2, gamma = 0.5)

    for _ in range(5):
        x = torch.randn(6, 8)
        m_ref(x).pow(2).mean().backward()
        o_ref.step()
        o_ref.zero_grad()
        sched_ref.step()

        g = o_pop.autograd(pop_noop(x).pow(2).mean(), flat = True)
        o_pop.step(grads = g)
        sched_pop.step()

    for p1, p2 in zip(m_ref.parameters(), m_pop.parameters()):
        assert torch.allclose(p1, p2, atol = 1e-6)
    assert pop_noop.slot('0.weight').state['steps'].item() == 5

    # state_dict save -> step -> reload restores exact trajectory (cache invalidated!)
    snap = deepcopy(pop_noop.state_dict())
    x_chk = torch.randn(6, 8)
    pop_noop(x_chk).pow(2).mean().backward()
    o_pop.step()
    o_pop.zero_grad()
    after_step = m_pop[0].weight.detach().clone()

    pop_noop.load_state_dict(snap)
    assert pop_noop.slot('0.weight').state['steps'].item() == 5
    pop_noop(x_chk).pow(2).mean().backward()
    o_pop.step()
    o_pop.zero_grad()
    assert torch.allclose(m_pop[0].weight, after_step, atol = 1e-6)

    # dtype sync on noop net (.double())
    m_pop.double()
    x64 = torch.randn(6, 8, dtype = torch.float64)
    g64 = o_pop.autograd(pop_noop(x64).pow(2).mean(), flat = True)
    o_pop.step(grads = g64)
    assert pop_noop.slot('0.weight').state['exp_avg'].dtype == torch.float64

    # noop -> populate(3) preserves py_type metadata & continues stepping
    m_pop.float()
    pop_noop.populate(3, init = 'clone')
    o_pop.step(grads = o_pop.autograd(pop_noop(torch.randn(6, 8)).pow(2).mean(), flat = True))
    assert (pop_noop.slot('0.weight').state['steps'] == 8).all()

    # 2. populated parity (uniform batched + post-reset hetero fallback + extract + share)
    refs = [mlp() for _ in range(3)]
    ref_opts = [PyScalarStepOpt(m.parameters(), lr = 2e-2) for m in refs]
    pop = Population([deepcopy(m) for m in refs])
    opt = pop.optimizer(PyScalarStepOpt, lr = 2e-2)
    assert opt._can_batch and not opt._can_batch_hetero

    for _ in range(4):
        x = torch.randn(3, 6, 8)
        for i, (m, o) in enumerate(zip(refs, ref_opts)):
            m(x[i]).pow(2).mean().backward()
            o.step()
            o.zero_grad()
        g = opt.autograd(pop(x, per_individual = True).pow(2).mean(dim = (1, 2)).sum(), flat = True)
        opt.step(grads = g)

    for i, m in enumerate(refs):
        for p1, p2 in zip(m.parameters(), pop[i].parameters()):
            assert torch.allclose(p1, p2, atol = 1e-6)

    # partial reset creates heterogeneous step counts (0 vs 4) -> clean per-member fallback
    opt.reset(0)
    ref_opts[0] = PyScalarStepOpt(refs[0].parameters(), lr = 2e-2)
    for _ in range(3):
        x = torch.randn(3, 6, 8)
        for i, (m, o) in enumerate(zip(refs, ref_opts)):
            m(x[i]).pow(2).mean().backward()
            o.step()
            o.zero_grad()
        pop(x, per_individual = True).pow(2).mean(dim = (1, 2)).sum().backward()
        opt.step()
        opt.zero_grad()

    for i, m in enumerate(refs):
        for p1, p2 in zip(m.parameters(), pop[i].parameters()):
            assert torch.allclose(p1, p2, atol = 1e-6)

    # extract returns native Python int/float state & continues with exact parity
    ind, ind_opt = pop.extract(1)
    first_p = next(ind.parameters())
    assert type(ind_opt.state[first_p]['steps']) is int
    assert type(ind_opt.state[first_p]['scale']) is float
    x1 = torch.randn(6, 8)
    ind(x1).pow(2).mean().backward()
    ind_opt.step()
    refs[1](x1).pow(2).mean().backward()
    ref_opts[1].step()
    for p1, p2 in zip(refs[1].parameters(), ind.parameters()):
        assert torch.allclose(p1, p2, atol = 1e-6)

    # share / unshare works with py_type aux state
    opt.share(reduce = 'max')
    opt.step(grads = opt.autograd(pop(torch.randn(6, 8)).pow(2).mean(), flat = True))
    opt.unshare()
    opt.step(grads = opt.autograd(pop(torch.randn(6, 8)).pow(2).mean(), flat = True))

    # 3. out-of-place state tensor replacement in noop mode stays synced with store
    m_oop_ref, m_oop = deepcopy(base), deepcopy(base)
    o_oop_ref = OutOfPlaceStateOpt(m_oop_ref.parameters(), lr = 2e-2)
    pop_oop = Population(m_oop)
    o_oop = pop_oop.optimizer(OutOfPlaceStateOpt, lr = 2e-2)
    for _ in range(4):
        x = torch.randn(6, 8)
        m_oop_ref(x).pow(2).mean().backward()
        o_oop_ref.step()
        o_oop_ref.zero_grad()
        o_oop.step(grads = o_oop.autograd(pop_oop(x).pow(2).mean(), flat = True))
    for p1, p2 in zip(m_oop_ref.parameters(), m_oop.parameters()):
        assert torch.allclose(p1, p2, atol = 1e-6)
    assert pop_oop.slot('0.weight').state['buf'].norm() > 0


def test_multi_loss_flat_autograd_damper_noop_and_populated():
    torch.manual_seed(20)

    def damper(g_r: torch.Tensor, g_t: torch.Tensor, eta: float = 0.5) -> torch.Tensor:
        n_r = g_r.norm(dim = -1, keepdim = True).clamp(min = 1e-8)
        n_t = g_t.norm(dim = -1, keepdim = True).clamp(min = 1e-8)
        dot = (g_r * g_t).sum(dim = -1, keepdim = True).clamp(max = 0.)
        g_t_proj = g_t - (dot / n_r.square()) * g_r
        weight = (n_t / n_r).clamp(max = 1.).pow(1. - eta)
        return g_r + (n_r / n_t) * weight * g_t_proj

    s, s_next = torch.randn(8, 8), torch.randn(8, 8)

    # identical code works in noop (D,) and populated (P, D)
    for pop_size in (None, 3):
        pop = Population(mlp(), pop_size = pop_size)
        opt = pop.optimizer(torch.optim.AdamW, lr = 1e-2)

        y, y_next = pop(s), pop(s_next)
        loss_r = y.pow(2).mean()
        loss_t = (y_next - y).pow(2).mean()

        g_r, g_t = opt.autograd(loss_r, loss_t, flat = True)
        expected_ndim = 1 if pop_size is None else 2
        assert g_r.ndim == expected_ndim and g_r.shape == pop.population_parameters.flat_params.shape

        # Theorem 1: g_damper . g_r >= ||g_r||^2
        g = damper(g_r, g_t, eta = 0.5)
        assert ((g * g_r).sum(dim = -1) >= g_r.pow(2).sum(dim = -1) - 1e-5).all()
        opt.step(grads = g)

        # sequence form always returns a tuple (even for length 1)
        (g_single,) = pop.autograd([pop(s).pow(2).mean()], flat = True)
        assert g_single.shape == g_r.shape

        # detached/zero-grad loss returns exact zeros; scoped names zero non-target params
        g_zero = opt.autograd(torch.tensor(0.), flat = True)
        assert g_zero.shape == g_r.shape and g_zero.abs().sum() == 0

        g_scoped = opt.autograd(pop(s).pow(2).mean(), names = ('2.bias',), flat = True)
        unpacked = pop.population_parameters.unpack_flat(g_scoped)
        assert unpacked['0.weight'].abs().sum() == 0 and unpacked['2.bias'].abs().sum() > 0

    # differentiable multi-loss flat autograd + functional_step
    pop = Population(mlp(), pop_size = 2)
    opt = pop.optimizer(torch.optim.Adam, lr = 1e-2)
    eta = nn.Parameter(torch.tensor(0.5))
    pp = pop.population_parameters
    y, y_next = pop(s, params = pp), pop(s_next, params = pp)
    g_r, g_t = opt.autograd(y.pow(2).mean(), (y_next - y).pow(2).mean(), params = pp, flat = True, create_graph = True)
    next_pp = opt.functional_step(grads = damper(g_r, g_t, eta = eta), params = pp, create_graph = True)
    pop(s, params = next_pp).pow(2).mean().backward()
    assert eta.grad is not None and eta.grad.isfinite()


def test_multi_optimizer_unrolled_backprop_and_ltm_consolidation():
    torch.manual_seed(25)
    make_double = lambda: mlp().double()
    x_tr = torch.randn(4, 6, 8, dtype = torch.float64)
    x_val = torch.randn(4, 6, 8, dtype = torch.float64)

    pop = Population(make_double, pop_size = 4)
    ltm = pop.clone(include_opt = False)
    ltm_opt = ltm.optimizer(torch.optim.AdamW, lr = 1e-1)
    ltm_solo = Population(pop[0])

    opt_sgd   = pop.optimizer(torch.optim.SGD, lr = 5e-2, momentum = 0.9, name = 'sgd')
    opt_rms   = pop.optimizer(torch.optim.RMSprop, lr = 1e-2, name = 'rms')
    opt_adamw = pop.optimizer(torch.optim.AdamW, lr = 1e-2, name = 'adamw')
    meta_target = nn.Parameter(torch.randn(1, 1, 4, dtype = torch.float64))

    # 1. Exact finite-difference check through SGD(momentum=0.9) -> RMSprop -> AdamW
    def unroll(target_val, *, create_graph = True):
        fast = pop.population_parameters
        for step_opt in (opt_sgd, opt_rms, opt_adamw):
            loss = (pop(x_tr, params = fast, per_individual = True) - target_val).pow(2).mean()
            fast = step_opt.functional_step(loss, fast, create_graph = create_graph, detach_preconditioner = False)
        return pop(x_val, params = fast, per_individual = True).pow(2).mean(), fast

    meta_loss, fast = unroll(meta_target, create_graph = True)
    meta_loss.backward()
    g_meta = meta_target.grad.clone()

    eps = 1e-6
    t_plus, t_minus = meta_target.detach().clone(), meta_target.detach().clone()
    t_plus[0, 0, 0] += eps
    t_minus[0, 0, 0] -= eps
    fd = (unroll(t_plus, create_graph = False)[0] - unroll(t_minus, create_graph = False)[0]) / (2 * eps)
    assert ((g_meta[0, 0, 0] - fd).abs() / fd.abs().clamp(min = 1e-12)) < 1e-4

    # 2. LTM consolidation for an individual via view lerp_, standalone lerp_, Reptile outer opt, and colocated ema_
    w0_before = ltm.view(0).flat_params.detach().clone()
    w1_before = ltm.view(1).flat_params.detach().clone()
    ltm.view(0).lerp_(fast, 0.1)
    assert torch.allclose(ltm.view(0).flat_params, w0_before.lerp(fast.flat_params[0].detach(), 0.1), atol = 1e-12)
    assert torch.equal(ltm.view(1).flat_params, w1_before)

    w0_solo = ltm_solo.flat_params.detach().clone()
    ltm_solo.lerp_(fast[0], 0.1)
    assert torch.allclose(ltm_solo.flat_params, w0_solo.lerp(fast.flat_params[0].detach(), 0.1), atol = 1e-12)

    # cross-member view lerp_ (individual 2 -> individual 0's LTM)
    w0_mid = ltm.view(0).flat_params.detach().clone()
    ltm.view(0).lerp_(pop.view(2), 0.25)
    assert torch.allclose(ltm.view(0).flat_params, w0_mid.lerp(pop.view(2).flat_params.detach(), 0.25), atol = 1e-12)

    w2_before = ltm.view(2).flat_params.detach().clone()
    ltm_opt.step(grads = ltm.flat_params - fast.flat_params.detach(), members = 2)
    assert not torch.allclose(ltm.view(2).flat_params, w2_before)
    assert torch.equal(ltm.view(1).flat_params, w1_before)

    ltm_coloc = pop.view(0).ema_('ltm', fast.flat_params[0], decay = 0.9)
    assert torch.allclose(ltm_coloc, fast.flat_params[0].detach())
    assert pop(x_val[0], params = pop.view(0).tracked('ltm')).shape == (6, 4)

    # 3. Heterogeneous optimizers per member in one inner loop (0,1=AdamW; 2,3=RMSprop)
    meta_target.grad = None
    pop.zero_grad()
    fast_h = pop.population_parameters
    for _ in range(2):
        l_a = (pop(x_tr, params = fast_h, per_individual = True) - meta_target).pow(2).mean()
        fast_h = opt_adamw.functional_step(l_a, fast_h, members = (0, 1), create_graph = True)
        l_b = (pop(x_tr, params = fast_h, per_individual = True) - meta_target).pow(2).mean()
        fast_h = opt_rms.functional_step(l_b, fast_h, members = (2, 3), create_graph = True)
    pop(x_val, params = fast_h, per_individual = True).pow(2).mean().backward()
    assert meta_target.grad.isfinite().all() and meta_target.grad.abs().sum() > 0
    assert (pop.population_parameters._pop_param.grad.abs().sum(dim = -1) > 0).all()


@param('opt_cls,kw', [
    (torch.optim.Adam, dict(lr = 1e-2)),
    (torch.optim.AdamW, dict(lr = 1e-2, weight_decay = 1e-2)),
    (torch.optim.SGD, dict(lr = 5e-2, momentum = 0.9)),
    (torch.optim.RMSprop, dict(lr = 1e-2)),
])
def test_batched_subset_step_parity_and_functional(opt_cls, kw):
    torch.manual_seed(400)
    x = torch.randn(5, 6, 8)
    pop_fast = Population(mlp, pop_size = 5)
    pop_ref = pop_fast.clone(include_opt = False)

    opt_fast = pop_fast.optimizer(opt_cls, **kw)
    opt_ref = pop_ref.optimizer(opt_cls, per_individual = True, **kw)
    assert opt_fast._can_batch

    # guard that batched path actually fires on subset steps (never falls back to _step_group)
    orig_step_group = opt_fast._step_group
    def forbid_fallback(*args, **kwargs):
        raise AssertionError('subset step unexpectedly fell back to _step_group')
    opt_fast._step_group = forbid_fallback

    # 1. uniform fresh subset, uniform warm subset, size-1 subset, reset + heterogeneous subset
    schedules = [
        ([0, 2, 4], None),
        ([0, 2, 4], None),
        ([3], None),
        ([1, 3], 'subset_flat'),
        ([0, 1, 2], 'full_flat'),
        ([2, 4], 'mapping'),
    ]
    for step_idx, (members, grad_mode) in enumerate(schedules):
        if step_idx == 4:
            # reset member 0 so [0, 1, 2] has heterogeneous step counts
            opt_fast.reset(0)
            opt_ref.reset(0)

        for p, o in ((pop_fast, opt_fast), (pop_ref, opt_ref)):
            o.zero_grad()
            if grad_mode is None:
                p(x[members], members = members, per_individual = True).pow(2).mean().backward()
                o.step(members = members)
            elif grad_mode == 'subset_flat':
                g = torch.full((len(members), p.flat_params.shape[-1]), 0.1 * (step_idx + 1))
                o.step(grads = g, members = members)
            elif grad_mode == 'full_flat':
                g = torch.arange(1, 6, dtype = torch.float32)[:, None].expand_as(p.flat_params) * 0.05
                o.step(grads = g, members = members)
            else:
                g_map = {k: torch.full_like(v, 0.05) for k, v in p.params_dict.items()}
                o.step(grads = g_map, members = members)

        assert torch.allclose(pop_fast.flat_params, pop_ref.flat_params, atol = 1e-5, rtol = 1e-4)

    # 2. single-member 1D flat grad
    g1 = torch.full((pop_fast.flat_params.shape[-1],), 0.03)
    opt_fast.step(grads = g1, members = 2)
    opt_ref.step(grads = g1, members = 2)
    assert torch.allclose(pop_fast.flat_params, pop_ref.flat_params, atol = 1e-5, rtol = 1e-4)

    # 3. differentiable functional_step on subset members hits batched path with exact meta-grad parity
    x_tr = torch.randn(5, 6, 8)
    target = torch.randn(5, 6, 4, requires_grad = True)
    sub_members = [1, 3]

    def run_meta(pop_obj, opt_obj, tgt):
        pp = pop_obj.population_parameters
        for _ in range(2):
            loss = (pop_obj(x_tr, params = pp, per_individual = True) - tgt).pow(2).mean()
            pp = opt_obj.functional_step(loss, pp, members = sub_members, create_graph = True)
        return pop_obj(x, params = pp, per_individual = True).pow(2).mean(), pp

    t_fast = target.detach().clone().requires_grad_(True)
    t_ref = target.detach().clone().requires_grad_(True)
    loss_f, pp_f = run_meta(pop_fast, opt_fast, t_fast)
    loss_r, pp_r = run_meta(pop_ref, opt_ref, t_ref)
    loss_f.backward()
    loss_r.backward()

    assert torch.allclose(pp_f.flat_params, pp_r.flat_params, atol = 1e-5, rtol = 1e-4)
    assert torch.allclose(t_fast.grad, t_ref.grad, atol = 1e-5, rtol = 1e-4)

    # 4. shared optimizer state + subset step parity
    opt_fast._step_group = orig_step_group
    opt_fast.share()
    opt_ref.share()
    opt_fast._step_group = forbid_fallback
    for p, o in ((pop_fast, opt_fast), (pop_ref, opt_ref)):
        o.zero_grad()
        p(x[[0, 2]], members = [0, 2], per_individual = True).pow(2).mean().backward()
        o.step(members = [0, 2])
    assert torch.allclose(pop_fast.flat_params, pop_ref.flat_params, atol = 1e-5, rtol = 1e-4)


def test_replace_preserves_optimizer_state_hps_and_tracks():
    torch.manual_seed(401)
    x = torch.randn(6, 8)
    pop = Population(mlp, pop_size = 4)
    opt = pop.optimizer(torch.optim.AdamW, lr = [1e-4, 5e-4, 2e-3, 1e-2])
    pop.track_('age', torch.tensor([10, 20, 30, 40]))

    # warm up member 0 and member 2 differently; leave member 3 unstepped
    for _ in range(3):
        opt.zero_grad()
        pop.view(0)(x).pow(2).mean().backward()
        opt.step(members = 0)
    opt.zero_grad()
    pop.view(2)(x).pow(2).mean().backward()
    opt.step(members = 2)

    # replace(3, 0, reset_opt=False) copies weights, opt state, _init, per-member lr, and tracked tensors
    pop.replace(3, 0, reset_opt = False)
    assert torch.equal(pop.flat_params[3], pop.flat_params[0])
    assert torch.equal(pop.slot('0.weight').state['exp_avg'][3], pop.slot('0.weight').state['exp_avg'][0])
    assert torch.equal(pop.slot('0.weight').state['step'][3], pop.slot('0.weight').state['step'][0])
    assert pop.opt.param_groups[0]['lr'][3].item() == pytest.approx(1e-4)
    assert pop.tracked('age')[3].item() == 10

    # subsequent identical gradients keep member 3 and member 0 identical
    g = torch.randn_like(pop.flat_params[0])
    opt.step(grads = torch.stack((g, g)), members = [0, 3])
    assert torch.allclose(pop.flat_params[3], pop.flat_params[0], atol = 1e-6)

    # view.copy_from_(other_view, include_opt=True) also copies opt state & tracked metadata
    pop.view(1).copy_from_(pop.view(2), include_opt = True)
    assert torch.equal(pop.flat_params[1], pop.flat_params[2])
    assert torch.equal(pop.slot('0.weight').state['step'][1], pop.slot('0.weight').state['step'][2])
    assert pop.opt.param_groups[0]['lr'][1].item() == pytest.approx(2e-3)
    assert pop.tracked('age')[1].item() == 30

    # reset_opt=True still resets opt state and tracked metadata
    pop.replace(3, 0, reset_opt = True)
    assert pop.slot('0.weight').state['step'][3].item() == 0.
    assert pop.slot('0.weight').state['exp_avg'][3].abs().sum() == 0.
    assert pop.tracked('age')[3].item() == 0
