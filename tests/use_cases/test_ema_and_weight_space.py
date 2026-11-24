from copy import deepcopy

import pytest
import torch
from torch import nn

from torch_population import Population


def make_mlp(dim_in = 8, dim_hidden = 16, dim_out = 4):
    return nn.Sequential(nn.Linear(dim_in, dim_hidden), nn.SiLU(), nn.Linear(dim_hidden, dim_out))


class NormFrozenNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(8, 16), nn.BatchNorm1d(16), nn.SiLU())
        self.head = nn.Linear(16, 4)
        self.frozen_scale = nn.Parameter(torch.ones(4), requires_grad = False)

    def forward(self, x):
        return self.head(self.encoder(x)) * self.frozen_scale


def test_parameter_ema_batchnorm_multi_timescale_and_empty_mask():
    torch.manual_seed(300)
    x = torch.randn(6, 8)

    # 1. BatchNorm (float running_mean/var + int64 num_batches_tracked) + frozen param + shared encoder
    for pop_size in (None, 4):
        pop = Population(NormFrozenNet, pop_size = pop_size, shared = ('encoder',) if pop_size else ())
        opt = pop.optimizer(torch.optim.AdamW, lr = 1e-2)
        ema = pop.clone(include_opt = False)

        pop(x).pow(2).mean().backward()
        opt.step()
        opt.zero_grad()

        if pop_size is None:
            w0 = ema.flat_params.detach().clone()
            ema.lerp_(pop, 0.2)
            assert torch.allclose(ema.flat_params, w0.lerp(pop.flat_params.detach(), 0.2), atol = 1e-6)
        else:
            # multi-timescale EMA including boundary rates 0.0 (frozen) and 1.0 (hard copy)
            w0 = ema.flat_params.detach().clone()
            taus = torch.tensor([0.0, 0.25, 0.5, 1.0])
            ema.lerp_(pop, taus)
            assert torch.equal(ema.flat_params[0], w0[0])
            assert torch.allclose(ema.flat_params[1], w0[1].lerp(pop.flat_params[1].detach(), 0.25), atol = 1e-6)
            assert torch.allclose(ema.flat_params[3], pop.flat_params[3].detach(), atol = 1e-6)

            # empty mask is a clean no-op
            before_empty = ema.flat_params.detach().clone()
            ema.lerp_(pop, 0.5, members = torch.zeros(pop_size, dtype = torch.bool))
            assert torch.equal(ema.flat_params, before_empty)

        # int64 num_batches_tracked stayed integer and forward succeeds
        assert ema.buffers_dict['encoder.1.num_batches_tracked'].dtype == torch.int64
        assert ema(x).isfinite().all()

        # colocated flat parameter EMA runnable directly via params=p_ema (with shared + frozen + BN!)
        p_ema = pop.ema_('param', pop.flat_params, decay = 0.99)
        assert pop(x, params = p_ema).shape == pop(x).shape


def test_gradient_ema_fisher_variance_and_full_lifecycle():
    torch.manual_seed(301)
    x = torch.randn(6, 8)

    pop = Population(make_mlp, pop_size = 4)
    opt = pop.optimizer(torch.optim.AdamW, lr = 1e-2)

    # online Fisher diagonal & gradient variance / SNR via ema_
    for _ in range(3):
        g = opt.autograd(pop(x).pow(2).mean(dim = (1, 2)).sum(), flat = True)
        g_mean = opt.ema_('g_mean', g, decay = 0.9)
        fisher = opt.ema_('fisher', g.square(), decay = 0.9)
        g_var = (fisher - g_mean.square()).clamp(min = 0.)
        assert (fisher >= 0).all() and g_var.isfinite().all()
        opt.step(grads = g)

    # edge cases: empty members mask, subset update, per-member decay, reset_, append, delete, slice, state_dict
    decays = torch.tensor([0.0, 0.5, 0.9, 1.0])
    pop.ema_('stat', torch.ones(4, 3), decay = decays)
    pop.ema_('stat', torch.full((4, 3), 5.), decay = decays, members = torch.zeros(4, dtype = torch.bool))
    assert torch.allclose(pop.tracked('stat'), torch.ones(4, 3))

    pop.ema_('stat', torch.full((4, 3), 5.), decay = decays)
    assert torch.allclose(pop.tracked('stat')[0], torch.full((3,), 5.))   # decay=0.0 -> overwrite
    assert torch.allclose(pop.tracked('stat')[1], torch.full((3,), 3.))   # decay=0.5 -> midpoint
    assert torch.allclose(pop.tracked('stat')[3], torch.ones(3))          # decay=1.0 -> frozen

    # reset_(1) + append(0): both initialize fresh on next ema_ without zero-bias
    pop.reset_(1)
    pop.delete(2)
    idx = pop.append(0)
    assert idx == 3 and pop.tracked('stat').shape == (4, 3)
    pop.ema_('stat', torch.full((4, 3), 9.), decay = 0.9)
    assert torch.allclose(pop.tracked('stat')[1], torch.full((3,), 9.))
    assert torch.allclose(pop.tracked('stat')[3], torch.full((3,), 9.))

    # state_dict persistence
    sd = {k: v.clone() for k, v in pop.state_dict().items()}
    pop2 = pop.clone(include_opt = False)
    pop2.clear_tracked()
    assert pop2.tracked('stat') is None
    pop.load_state_dict(sd)
    assert torch.allclose(pop.tracked('stat')[3], torch.full((3,), 9.))


@pytest.mark.parametrize('pop_size', (None, 3))
def test_sam_exact_parity_and_zero_grad_edge_case(pop_size):
    torch.manual_seed(302)
    x = torch.randn(6, 8)
    rho = 0.05

    pop = Population(NormFrozenNet, pop_size = pop_size, shared = ('encoder',) if pop_size else ())
    pop.eval()  # freeze BN stats so ref clone comparison is exact
    opt = pop.optimizer(torch.optim.SGD, lr = 1e-2)

    g = opt.autograd(pop(x).pow(2).mean(), flat = True)
    eps = rho * g / g.norm(dim = -1, keepdim = True).clamp(min = 1e-8)
    g_sam = opt.autograd(pop(x, params = pop.flat_params + eps.detach()).pow(2).mean(), flat = True)

    ref = pop.clone(include_opt = False)
    with torch.no_grad():
        if pop_size is None:
            for p, e in zip((p for p in ref.unwrap().parameters() if p.requires_grad), ref.population_parameters.unpack_flat(eps).values()):
                p.add_(e)
        else:
            ref.population_parameters._pop_param.add_(eps)
    g_ref = ref.autograd(ref(x).pow(2).mean(), flat = True)
    assert torch.allclose(g_sam, g_ref, atol = 1e-5)

    # zero-grad edge case (detached loss -> eps = 0 -> no NaN)
    g0 = opt.autograd(torch.tensor(0.), flat = True)
    eps0 = rho * g0 / g0.norm(dim = -1, keepdim = True).clamp(min = 1e-8)
    assert eps0.isfinite().all() and eps0.abs().sum() == 0
    opt.step(grads = g_sam)


def test_evolution_strategies_antithetic_on_single_noop_model():
    torch.manual_seed(303)
    net = NormFrozenNet().eval()
    pop = Population(net)
    opt = pop.optimizer(torch.optim.AdamW, lr = 1e-2)

    x = torch.randn(6, 8)
    D = pop.flat_params.numel()
    sigma = 0.05
    half = torch.randn(8, D)
    noise = torch.cat((half, -half), dim = 0)  # 16 antithetic perturbations on 1 noop model
    thetas = pop.flat_params.detach() + sigma * noise

    outs = pop(x, params = thetas)
    assert outs.shape == (16, 6, 4)

    # exact parity with manual forward on perturbed clones (including BN buffers & frozen_scale!)
    for k in (0, 7, 15):
        m = deepcopy(net)
        with torch.no_grad():
            for p, val in zip((p for p in m.parameters() if p.requires_grad), pop.population_parameters.unpack_flat(thetas[k]).values()):
                p.copy_(val)
        assert torch.allclose(outs[k], m(x), atol = 1e-5)

    rewards = -outs.pow(2).mean(dim = (1, 2))
    g_es = -(rewards[:, None] * noise).mean(dim = 0) / sigma
    w_before = pop.flat_params.detach().clone()
    opt.step(grads = g_es)
    assert not torch.allclose(pop.flat_params, w_before)


def test_differentiable_model_soup_and_mode_connectivity():
    torch.manual_seed(304)
    pop = Population(make_mlp, pop_size = 4)
    x = torch.randn(6, 8)

    # learnable convex soup weights (flows gradients into BOTH soup weights AND all P members!)
    logits = nn.Parameter(torch.randn(4))
    weights = logits.softmax(dim = 0)
    soup_theta = weights @ pop.flat_params  # (D,)

    soup_out = pop(x, params = soup_theta)
    assert soup_out.shape == (6, 4)
    assert torch.allclose(soup_out, pop.mean(weights = weights.detach())(x), atol = 1e-5)

    soup_out.pow(2).mean().backward()
    assert logits.grad is not None and logits.grad.isfinite().all() and logits.grad.abs().sum() > 0
    assert (pop.population_parameters._pop_param.grad.abs().sum(dim = -1) > 0).all()

    # linear mode connectivity barrier along alpha in [0, 1] between member 0 and member 1 in one vmapped call
    alphas = torch.linspace(0., 1., 5)[:, None]
    path = (1. - alphas) * pop.flat_params[0] + alphas * pop.flat_params[1]  # (5, D)
    path_out = pop(x, params = path)
    assert path_out.shape == (5, 6, 4)
    assert torch.allclose(path_out[0], pop[0](x), atol = 1e-5)
    assert torch.allclose(path_out[-1], pop[1](x), atol = 1e-5)
