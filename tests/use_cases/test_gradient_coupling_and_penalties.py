import math
from copy import deepcopy

import pytest
import torch
from torch import nn

from torch_population import Population


def make_mlp(dim_in = 8, dim_hidden = 16, dim_out = 4):
    return nn.Sequential(nn.Linear(dim_in, dim_hidden), nn.SiLU(), nn.Linear(dim_hidden, dim_out))


def test_gradient_swap_exact_parity_and_general_mixing():
    torch.manual_seed(200)
    m0, m1 = make_mlp(), make_mlp()

    # 1. Exact bit-for-bit parity of 2-member grad swap (both autograd+step and reconcile+backward)
    r0, r1 = deepcopy(m0), deepcopy(m1)
    ro0 = torch.optim.AdamW(r0.parameters(), lr = 1e-2)
    ro1 = torch.optim.AdamW(r1.parameters(), lr = 1e-2)

    pop = Population([deepcopy(m0), deepcopy(m1)])
    opt = pop.optimizer(torch.optim.AdamW, lr = 1e-2)

    pop_rec = Population([deepcopy(m0), deepcopy(m1)])
    opt_rec = pop_rec.optimizer(torch.optim.AdamW, lr = 1e-2)

    for _ in range(5):
        x, y = torch.randn(12, 8), torch.randn(12, 4)

        (r0(x) - y).pow(2).mean().backward()
        (r1(x) - y).pow(2).mean().backward()
        for p0, p1 in zip(r0.parameters(), r1.parameters()):
            p0.grad, p1.grad = p1.grad.clone(), p0.grad.clone()
        ro0.step(); ro1.step()
        ro0.zero_grad(); ro1.zero_grad()

        g = opt.autograd((pop(x) - y).pow(2).mean(dim = (1, 2)).sum(), flat = True)
        opt.step(grads = g.flip(0))

        opt_rec.zero_grad()
        pop_rec.reconcile((pop_rec(x) - y).pow(2).mean(dim = (1, 2)).sum(), fn = lambda gr: gr.flip(0)).backward()
        opt_rec.step()

    for i, ref in enumerate((r0, r1)):
        for p_ref, p_pop, p_rec in zip(ref.parameters(), pop[i].parameters(), pop_rec[i].parameters()):
            assert torch.allclose(p_ref, p_pop, atol = 1e-6)
            assert torch.allclose(p_ref, p_rec, atol = 1e-6)

    # 2. Edge cases: P=1 flip is identity; P=4 ring shift, subset swap, EMA swap, and meta-learned mixing matrix M
    pop1 = Population(make_mlp(), pop_size = 1)
    opt1 = pop1.optimizer(torch.optim.SGD, lr = 1e-2)
    x = torch.randn(6, 8)
    g1 = opt1.autograd(pop1(x).pow(2).mean(), flat = True)
    assert torch.equal(g1, g1.flip(0))
    opt1.step(grads = g1.flip(0))

    pop4 = Population(make_mlp, pop_size = 4)
    opt4 = pop4.optimizer(torch.optim.AdamW, lr = 1e-2)
    g4 = opt4.autograd(pop4(x).pow(2).mean(dim = (1, 2)).sum(), flat = True)
    opt4.step(grads = g4.roll(1, dims = 0))                         # ring swap across all 4
    opt4.step(grads = opt4.ema_('g', g4, decay = 0.9).flip(0))      # swap gradient EMAs

    # subset swap: only swap members (1, 3), leave (0, 2) untouched
    w02_before = pop4.flat_params[[0, 2]].detach().clone()
    g_sub = opt4.autograd(pop4(x).pow(2).mean(dim = (1, 2)).sum(), flat = True)[[3, 1]]
    opt4.step(grads = g_sub, members = (1, 3))
    assert torch.equal(pop4.flat_params[[0, 2]], w02_before)

    # differentiable meta-learned (P, P) gradient mixing matrix M
    mix_logits = nn.Parameter(torch.zeros(4, 4))
    pp = pop4.population_parameters
    g_diff = opt4.autograd(pop4(x, params = pp).pow(2).mean(dim = (1, 2)).sum(), params = pp, flat = True, create_graph = True)
    M = mix_logits.softmax(dim = -1)
    next_pp = opt4.functional_step(grads = M @ g_diff, params = pp, create_graph = True)
    pop4(x, params = next_pp).pow(2).mean().backward()
    assert mix_logits.grad is not None and mix_logits.grad.isfinite().all() and mix_logits.grad.abs().sum() > 0


@pytest.mark.parametrize('pop_size', (None, 3))
def test_parameter_gradient_penalty_exact_parity(pop_size):
    torch.manual_seed(201)
    x = torch.randn(6, 8)

    if pop_size is None:
        m_ref = make_mlp(dim_out = 1)
        m_pop = deepcopy(m_ref)
        pop = Population(m_pop)

        l_ref = m_ref(x).pow(2).mean()
        gs_ref = torch.autograd.grad(l_ref, list(m_ref.parameters()), create_graph = True, retain_graph = True)
        (l_ref + 0.1 * sum(g.square().sum() for g in gs_ref)).backward()

        l_pop = pop(x).pow(2).mean()
        g_pop = pop.autograd(l_pop, flat = True, create_graph = True, retain_graph = True)
        (l_pop + 0.1 * g_pop.square().sum()).backward()

        for p_ref, p_pop in zip(m_ref.parameters(), m_pop.parameters()):
            assert torch.allclose(p_ref.grad, p_pop.grad, atol = 1e-5)
    else:
        refs = [make_mlp(dim_out = 1) for _ in range(pop_size)]
        pop = Population([deepcopy(m) for m in refs])

        l_pop = pop(x).pow(2).mean(dim = (1, 2)).sum()
        g_pop = pop.autograd(l_pop, flat = True, create_graph = True, retain_graph = True)
        (l_pop + 0.1 * g_pop.square().sum()).backward()

        for i, m in enumerate(refs):
            l_ref = m(x).pow(2).mean()
            gs_ref = torch.autograd.grad(l_ref, list(m.parameters()), create_graph = True, retain_graph = True)
            (l_ref + 0.1 * sum(g.square().sum() for g in gs_ref)).backward()
            for (n, p_ref), (_, p_pop_g) in zip(m.named_parameters(), pop.grads_dict.items()):
                assert torch.allclose(p_ref.grad, p_pop_g[i], atol = 1e-5), f'mismatch member {i} {n}'

    # edge case: shared layer + param-GP
    if pop_size is not None:
        pop_sh = Population(make_mlp(dim_out = 1), pop_size = pop_size, shared = ('0.bias',))
        l_sh = pop_sh(x).pow(2).mean()
        g_sh = pop_sh.autograd(l_sh, flat = True, create_graph = True, retain_graph = True)
        (l_sh + 0.1 * g_sh.square().sum()).backward()
        assert pop_sh.grads_dict['0.weight'].isfinite().all()
        assert pop_sh.grads_dict['0.bias'].isfinite().all()


@pytest.mark.parametrize('pop_size', (None, 3))
def test_input_gradient_penalty_r1_wgangp_and_reconcile(pop_size):
    torch.manual_seed(202)
    x = torch.randn(6, 8)

    # 1. Exact parity for R1 input-gradient penalty (||grad_x D(x)||^2)
    if pop_size is None:
        m_ref = make_mlp(dim_out = 1)
        m_pop = deepcopy(m_ref)
        pop = Population(m_pop)

        xr = x.detach().clone().requires_grad_(True)
        or_ = m_ref(xr)
        gxr = torch.autograd.grad(or_.sum(), xr, create_graph = True)[0]
        (or_.pow(2).mean() + 0.1 * gxr.square().sum(dim = -1).mean()).backward()

        xp = x.detach().clone().requires_grad_(True)
        op = pop(xp)
        gxp = torch.autograd.grad(op.sum(), xp, create_graph = True)[0]
        (op.pow(2).mean() + 0.1 * gxp.square().sum(dim = -1).mean()).backward()

        for p_ref, p_pop in zip(m_ref.parameters(), m_pop.parameters()):
            assert torch.allclose(p_ref.grad, p_pop.grad, atol = 1e-5)
    else:
        refs = [make_mlp(dim_out = 1) for _ in range(pop_size)]
        pop = Population([deepcopy(m) for m in refs])

        xp = x.unsqueeze(0).expand(pop_size, -1, -1).clone().requires_grad_(True)
        op = pop(xp, per_individual = True)
        gxp = torch.autograd.grad(op.sum(), xp, create_graph = True)[0]
        (op.pow(2).mean(dim = (1, 2)).sum() + 0.1 * gxp.square().sum(dim = -1).mean(dim = -1).sum()).backward()

        for i, m in enumerate(refs):
            xi = x.detach().clone().requires_grad_(True)
            oi = m(xi)
            gxi = torch.autograd.grad(oi.sum(), xi, create_graph = True)[0]
            (oi.pow(2).mean() + 0.1 * gxi.square().sum(dim = -1).mean()).backward()
            for (n, p_ref), (_, p_pop_g) in zip(m.named_parameters(), pop.grads_dict.items()):
                assert torch.allclose(p_ref.grad, p_pop_g[i], atol = 1e-5), f'mismatch member {i} {n}'

    # 2. Two-sided WGAN-GP (safe norm at zero grad) + reconciled against main loss via opt.autograd
    pop2 = Population(make_mlp(dim_out = 1), pop_size = pop_size)
    opt2 = pop2.optimizer(torch.optim.AdamW, lr = 1e-2)
    x_in = x.detach().clone().requires_grad_(True) if pop_size is None else x.unsqueeze(0).expand(pop_size, -1, -1).clone().requires_grad_(True)
    out2 = pop2(x_in) if pop_size is None else pop2(x_in, per_individual = True)
    gx = torch.autograd.grad(out2.sum(), x_in, create_graph = True)[0]
    wgan_gp = (gx.square().sum(dim = -1).add(1e-12).sqrt() - 1.).square().mean()
    g_main, g_gp = opt2.autograd(out2.pow(2).mean(), wgan_gp, flat = True)
    assert g_main.isfinite().all() and g_gp.isfinite().all()
    opt2.step(grads = g_main + 0.1 * g_gp)


def test_svgd_weight_space_repulsive_ensemble():
    torch.manual_seed(203)
    pop = Population(make_mlp, pop_size = 4, init = 'clone', noise_std = 1e-2)
    opt = pop.optimizer(torch.optim.Adam, lr = 1e-2)
    x, y = torch.randn(10, 8), torch.randn(10, 4)

    init_spread = torch.pdist(pop.flat_params.detach()).mean()
    for _ in range(5):
        g = opt.autograd((pop(x) - y).pow(2).mean(dim = (1, 2)).sum(), flat = True)
        theta = pop.flat_params.detach()
        dist_sq = torch.cdist(theta, theta).square()
        h = dist_sq.median().clamp(min = 1e-6) / math.log(len(pop) + 1)
        K = (-dist_sq / h).exp()
        # repulsive Stein variational gradient in weight space
        phi = (K @ g - (2. / h) * (K.sum(dim = 1, keepdim = True) * theta - K @ theta)) / len(pop)
        assert phi.isfinite().all()
        opt.step(grads = phi)

    assert torch.pdist(pop.flat_params.detach()).mean() > init_spread
