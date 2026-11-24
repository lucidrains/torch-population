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
