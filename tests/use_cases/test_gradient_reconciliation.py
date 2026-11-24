from functools import partial

import pytest
import torch
from torch import Tensor, nn

from torch_population import Population, PopulationParameters


def make_mlp(dim_in = 8, dim_hidden = 16, dim_out = 4):
    return nn.Sequential(nn.Linear(dim_in, dim_hidden), nn.SiLU(), nn.Linear(dim_hidden, dim_out))


def damper_grad(g_r: Tensor, g_t: Tensor, eta: float | Tensor = 0.5) -> Tensor:
    # DAMPER: Return-Prioritized Gradient Control for Smooth Policies (Ko, Goo & Hong, arXiv:2609.38903)
    n_r = g_r.norm(dim = -1, keepdim = True).clamp(min = 1e-8)
    n_t = g_t.norm(dim = -1, keepdim = True).clamp(min = 1e-8)
    g_t_proj = g_t - ((g_r * g_t).sum(dim = -1, keepdim = True).clamp(max = 0.) / n_r.square()) * g_r
    weight = (n_t / n_r).clamp(max = 1.).pow(1. - eta)
    return g_r + (n_r / n_t) * weight * g_t_proj


@pytest.mark.parametrize('pop_size', (None, 1, 4))
@pytest.mark.parametrize('eta', (0., 0.5, 1.))
def test_damper_theorem1_and_edge_cases(pop_size, eta):
    torch.manual_seed(100)
    pop = Population(make_mlp, pop_size = pop_size)
    opt = pop.optimizer(torch.optim.AdamW, lr = 1e-2)
    s, s_next = torch.randn(8, 8), torch.randn(8, 8)

    # 1. standard DAMPER update (identical code for noop, P=1, and P=4)
    y, y_next = pop(s), pop(s_next)
    loss_r = y.pow(2).mean()
    loss_t = (y_next - y).pow(2).mean()
    g_r, g_t = opt.autograd(loss_r, loss_t, flat = True)

    g = damper_grad(g_r, g_t, eta = eta)
    # Theorem 1: g_damper . g_r >= ||g_r||^2
    assert ((g * g_r).sum(dim = -1) >= g_r.pow(2).sum(dim = -1) - 1e-5).all()
    assert g.isfinite().all()
    opt.step(grads = g)

    # 2. degenerate edge cases: zero aux grad (s_next == s), zero primary grad, conflicting vs aligned
    g_zero_t = damper_grad(g_r, torch.zeros_like(g_t), eta = eta)
    assert torch.allclose(g_zero_t, g_r, atol = 1e-6)

    g_zero_r = damper_grad(torch.zeros_like(g_r), g_t, eta = eta)
    assert g_zero_r.isfinite().all()

    # strictly conflicting (g_t = -2 * g_r): projection annihilates g_t -> returns g_r exactly
    g_opp = damper_grad(g_r, -2. * g_r, eta = eta)
    assert torch.allclose(g_opp, g_r, atol = 1e-5)

    # strictly aligned (g_t = 0.5 * g_r): eta=0 keeps 0.5*g_r; eta=1 rescales to 1.0*g_r
    g_aligned = damper_grad(g_r, 0.5 * g_r, eta = eta)
    expected_scale = 1. + (0.5 ** (1. - eta))
    assert torch.allclose(g_aligned, expected_scale * g_r, atol = 1e-5)


def test_optimizer_agnostic_reconcile_multi_module_and_unused_branches():
    torch.manual_seed(101)

    class DualBranchActor(nn.Module):
        def __init__(self):
            super().__init__()
            self.motion_enc = nn.Linear(8, 8)
            self.vel_enc = nn.Linear(8, 8)
            self.head = nn.Linear(8, 4)

        def forward(self, x, pad, use_motion = True):
            enc = self.motion_enc if use_motion else self.vel_enc
            return self.head(torch.tanh(enc(x + pad)))

    class Agent(nn.Module):
        def __init__(self):
            super().__init__()
            self.actor = DualBranchActor()
            self.pad_embed = nn.Linear(8, 8)
            self.world_model = nn.Linear(4, 8)
            # spans actor + nested submodule (deduped) + standalone pad_embed + None; zero state_dict keys
            self.actor_params = PopulationParameters((self.actor, self.actor.head, self.pad_embed, None))

    agent = Agent()
    assert not any(k.startswith('actor_params.') for k in agent.state_dict())

    # user owns a plain PyTorch optimizer + gradient clipping + composite loss + gradient accumulation
    user_opt = torch.optim.AdamW(list(agent.actor.parameters()) + list(agent.pad_embed.parameters()), lr = 1e-2)
    for p in agent.world_model.parameters():
        p.requires_grad_(False)

    x, x_next = torch.randn(6, 8), torch.randn(6, 8)
    user_opt.zero_grad()

    # 2-step gradient accumulation with composite extra regularization
    for _ in range(2):
        pad = agent.pad_embed(torch.ones(6, 8))
        y, y_next = agent.actor(x, pad, use_motion = True), agent.actor(x_next, pad, use_motion = True)
        loss_r = agent.world_model(y).pow(2).mean()
        loss_t = (y_next - y).pow(2).mean()
        reconciled = agent.actor_params.reconcile(loss_r, loss_t, fn = partial(damper_grad, eta = 0.5))
        assert torch.equal(reconciled, loss_r.detach())
        ((reconciled + 0.05 * agent.pad_embed.bias.pow(2).sum()) / 2.).backward()

    assert agent.actor.motion_enc.weight.grad is not None and agent.actor.motion_enc.weight.grad.abs().sum() > 0
    assert agent.pad_embed.weight.grad is not None and agent.pad_embed.weight.grad.abs().sum() > 0
    assert agent.actor.vel_enc.weight.grad is None       # inactive branch stays None (no AdamW decay)
    assert agent.world_model.weight.grad is None          # frozen module stays None

    torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), 1.0)
    user_opt.step()

    # torch.no_grad() short-circuits cleanly
    with torch.no_grad():
        pad = agent.pad_embed(torch.ones(6, 8))
        y, y_next = agent.actor(x, pad), agent.actor(x_next, pad)
        out = agent.actor_params.reconcile(y.pow(2).mean(), (y_next - y).pow(2).mean(), fn = damper_grad)
        assert not out.requires_grad


@pytest.mark.parametrize('num_tasks', (1, 3))
def test_pcgrad_and_smoothed_grokfast_reconciliation(num_tasks):
    torch.manual_seed(102)
    pop = Population(make_mlp, pop_size = 3)
    opt = pop.optimizer(torch.optim.AdamW, lr = 1e-2)
    x = torch.randn(8, 8)
    targets = [torch.randn(8, 4) for _ in range(num_tasks)]

    def pcgrad(*grads: Tensor) -> Tensor:
        proj = [g.clone() for g in grads]
        for i in range(len(proj)):
            for j, gj in enumerate(grads):
                if i == j:
                    continue
                dot = (proj[i] * gj).sum(dim = -1, keepdim = True).clamp(max = 0.)
                proj[i] = proj[i] - (dot / gj.square().sum(dim = -1, keepdim = True).clamp(min = 1e-8)) * gj
        return sum(proj)

    # sequence form works identically for K=1 and K=3
    losses = [(pop(x) - t).pow(2).mean() for t in targets]
    grads = opt.autograd(losses, flat = True)
    assert isinstance(grads, tuple) and len(grads) == num_tasks

    g_pc = pcgrad(*grads)
    # Grokfast: amplify low-frequency gradient EMA (resets cleanly on member reset)
    g_ema = opt.ema_('grokfast', g_pc, decay = 0.9).clone()
    assert torch.allclose(g_ema, g_pc)
    opt.step(grads = g_pc + 2.0 * g_ema)

    # reset member 1 -> its grokfast EMA re-initializes fresh without zero-bias
    pop.reset_(1)
    assert pop.tracked('grokfast')[1].abs().sum() == 0
    losses2 = [(pop(x) - t).pow(2).mean() for t in targets]
    g2 = pcgrad(*opt.autograd(losses2, flat = True))
    g_ema2 = opt.ema_('grokfast', g2, decay = 0.9)
    assert torch.allclose(g_ema2[1], g2[1])
    assert torch.allclose(g_ema2[0], g_ema[0].lerp(g2[0], 0.1), atol = 1e-6)
