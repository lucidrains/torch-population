from copy import deepcopy

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.func import functional_call
from torch.nn import Module

from torch_population import Adam, AdamW, Population, PopulationParameters

param = pytest.mark.parametrize


class RichNet(Module):
    def __init__(self, dim_in = 8, dim_hidden = 16, dim_out = 4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(dim_in, dim_hidden),
            nn.SiLU(),
        )
        self.head = nn.Linear(dim_hidden, dim_out)
        self.register_buffer('scale', torch.tensor(1.25))
        self.register_buffer('temp', torch.tensor(0.5), persistent = False)

    def forward(self, x):
        return self.head(self.encoder(x)) * self.scale * self.temp


# 1. Physical colocation inside PopulationParameters

def test_population_parameters_colocates_everything_physically():
    torch.manual_seed(0)
    net = RichNet()
    pop = Population(net, pop_size = 4)
    opt = pop.adam(lr = 1e-2, shared = False)

    pp = pop.population_parameters
    assert isinstance(pp, PopulationParameters)

    total_trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    num_trainable_tensors = sum(1 for p in net.parameters() if p.requires_grad)

    # All population parameters, moments, variances, and steps live in single flat contiguous tensors
    assert pp.flat_params.shape == (4, total_trainable)
    assert pp.flat_params.is_contiguous()
    assert pp.flat_exp_avg.shape == (4, total_trainable)
    assert pp.flat_exp_avg.is_contiguous()
    assert pp.flat_exp_avg_sq.shape == (4, total_trainable)
    assert pp.flat_exp_avg_sq.is_contiguous()
    assert pp.flat_steps.shape == (4, num_trainable_tensors)
    assert pp.flat_steps.dtype == torch.int64

    # Structured dicts are zero-copy views directly into the colocated flat tensors
    named_p = pp.named_parameters_dict()
    named_opt = pp.named_optimizer_state()
    for name, p_view in named_p.items():
        assert p_view.untyped_storage().data_ptr() == pp.flat_params.untyped_storage().data_ptr()
        assert named_opt[name]['exp_avg'].untyped_storage().data_ptr() == pp.flat_exp_avg.untyped_storage().data_ptr()
        assert named_opt[name]['exp_avg_sq'].untyped_storage().data_ptr() == pp.flat_exp_avg_sq.untyped_storage().data_ptr()

    # In-place mutation of flat_params immediately reflects in forward and individual views
    x = torch.randn(5, 8)
    with torch.no_grad():
        pp.flat_params[1].zero_()
    assert torch.allclose(pop(x)[1], torch.zeros(5, 4))
    assert torch.equal(pp[1]['named_parameters']['head.weight'], torch.zeros(4, 16))

    # Shared optimizer state: flat_exp_avg / flat_exp_avg_sq have shape (D,), shared across all individuals
    opt.share()
    assert pp.flat_exp_avg.shape == (total_trainable,)
    assert pp.flat_exp_avg_sq.shape == (total_trainable,)
    assert pp.flat_steps.shape == (num_trainable_tensors,)

    ind0 = pp[0]
    ind3 = pp[3]
    assert ind0['named_optimizer_state']['head.weight']['exp_avg'].untyped_storage().data_ptr() == pp.flat_exp_avg.untyped_storage().data_ptr()
    assert ind3['named_optimizer_state']['head.weight']['exp_avg'].untyped_storage().data_ptr() == pp.flat_exp_avg.untyped_storage().data_ptr()


# 2. Single-individual dict matches PyTorch conventions & loads directly into native Module + torch.optim

@param('use_adamw', [False, True])
def test_individual_dict_matches_pytorch_and_loads_into_native_torch(use_adamw):
    torch.manual_seed(1)
    net = RichNet()
    pop = Population(net, pop_size = 4)
    if use_adamw:
        opt = pop.adamw(lr = 1e-2, betas = (0.9, 0.999), eps = 1e-8, weight_decay = 1e-2, shared = False)
    else:
        opt = pop.adam(lr = 1e-2, betas = (0.9, 0.999), eps = 1e-8, weight_decay = 0., shared = False)

    # Train 5 steps inside Population
    for _ in range(5):
        x_pop = torch.randn(4, 6, 8)
        y_pop = torch.randn(4, 6, 4)
        opt.zero_grad()
        (pop(x_pop, per_individual = True) - y_pop).pow(2).mean(dim = (1, 2)).sum().backward()
        opt.step()

    pp = pop.population_parameters
    ind_dict = pop.individual_dict(2)
    pp_dict = pp[2]
    assert set(ind_dict.keys()) == set(pp_dict.keys())

    # Matches PyTorch shapes and keys
    ref_named_p = dict(net.named_parameters())
    ref_named_b = dict(net.named_buffers())

    assert len(ind_dict['parameters']) == len(ref_named_p)
    assert list(ind_dict['named_parameters'].keys()) == list(ref_named_p.keys())
    for name, ref_p in ref_named_p.items():
        t = ind_dict['named_parameters'][name]
        assert t.shape == ref_p.shape
        assert t.is_contiguous()
        assert t.untyped_storage().data_ptr() == pp.flat_params.untyped_storage().data_ptr()

    assert list(ind_dict['named_buffers'].keys()) == list(ref_named_b.keys())
    assert list(ind_dict['state_dict'].keys()) == list(net.state_dict().keys())

    # functional_call directly with the single-individual dict matches pop(x)[2]
    x_test = torch.randn(5, 8)
    out_fc = functional_call(
        deepcopy(net),
        (ind_dict['named_parameters'], ind_dict['named_buffers']),
        (x_test,)
    )
    assert torch.allclose(out_fc, pop(x_test)[2], atol = 1e-6)

    # Load cloned PyTorch-compatible dict into a fresh native PyTorch network + native torch.optim.Adam/AdamW
    ind_cloned = pop.individual_dict(2, clone = True)
    fresh_net = RichNet()
    fresh_net.load_state_dict(ind_cloned['state_dict'])
    opt_cls = torch.optim.AdamW if use_adamw else torch.optim.Adam
    fresh_opt = opt_cls(fresh_net.parameters(), lr = 1e-2)
    fresh_opt.load_state_dict(ind_cloned['optimizer_state'])

    # Continue training both for 5 more steps on identical batches -> must match to < 1e-6
    for _ in range(5):
        x_pop = torch.randn(4, 6, 8)
        y_pop = torch.randn(4, 6, 4)

        opt.zero_grad()
        (pop(x_pop, per_individual = True) - y_pop).pow(2).mean(dim = (1, 2)).sum().backward()
        opt.step()

        fresh_opt.zero_grad()
        (fresh_net(x_pop[2]) - y_pop[2]).pow(2).mean().backward()
        fresh_opt.step()

    for name, p_ref in fresh_net.named_parameters():
        assert torch.allclose(pp[2]['named_parameters'][name], p_ref, atol = 1e-6)

    # Write a PyTorch-style dict back into slot 0 via pp[0] = ...
    pp[0] = {
        'named_parameters': dict(fresh_net.named_parameters()),
        'named_buffers': dict(fresh_net.named_buffers()),
        'optimizer_state': fresh_opt.state_dict(),
    }
    assert torch.allclose(pop(x_test)[0], fresh_net(x_test), atol = 1e-6)
    assert int(pp[0]['named_optimizer_state']['head.weight']['step'].item()) == 10


# 3. Population-level unified dict + noop mode compatibility

def test_population_as_dict_and_noop_mode():
    net = RichNet()
    pop_noop = Population(net)
    opt_noop = pop_noop.adam(lr = 1e-2)

    x = torch.randn(4, 8)
    opt_noop.zero_grad()
    pop_noop(x).pow(2).mean().backward()
    opt_noop.step()

    d_noop = pop_noop.individual_dict(0)
    assert list(d_noop['named_parameters'].keys()) == list(dict(net.named_parameters()).keys())
    assert d_noop['named_parameters']['head.weight'] is net.head.weight
    assert int(d_noop['named_optimizer_state']['head.weight']['step'].item()) == 1

    # Population-wide as_dict holds everything together
    pop = Population(net, pop_size = 3)
    pop.adam(lr = 1e-2, shared = False)
    pop_dict = pop.population_parameters.as_dict()
    assert pop_dict['named_parameters']['head.weight'].shape == (3, 4, 16)
    assert pop_dict['named_optimizer_state']['head.weight']['exp_avg'].shape == (3, 4, 16)
    assert pop_dict['named_optimizer_state']['head.weight']['step'].shape == (3,)
    assert len(pop_dict['optimizer_state']['param_groups'][0]['params']) == len(pop_dict['parameters'])


# 4. DiscoRL-style meta-learning loop: differentiable functional_step, detach, EMA, masked reset

@param('shared', [False, True])
def test_disco_rl_meta_learning_loop_simplified(shared):
    """End-to-end mini DiscoRL loop using Population + PopulationParameters + Adam.functional_step:
    - inner loop updates a population of policies differentiably through Adam
    - meta-loss on updated population backprops through inner updates into a meta-network
    - detach() truncates TBPTT in one call
    - reset_(mask) resets masked agents' params + optimizer state in one call
    - ema_pp.lerp_(pp, ...) tracks target policy parameters in one call
    """
    torch.manual_seed(42)

    class ToyPolicy(Module):
        def __init__(self, dim_obs = 6, dim_hidden = 16, num_actions = 3):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(dim_obs, dim_hidden),
                nn.SiLU(),
                nn.Linear(dim_hidden, num_actions),
            )

        def forward(self, obs):
            return self.net(obs)

    P = 6
    pop = Population(ToyPolicy(), pop_size = P)
    inner_opt = pop.adam(lr = 1e-2, shared = shared)

    # Meta-network learns target logits conditioned on observations
    meta_net = nn.Sequential(nn.Linear(6, 16), nn.SiLU(), nn.Linear(16, 3))
    meta_opt = torch.optim.Adam(meta_net.parameters(), lr = 1e-2)

    pp = pop.population_parameters
    ema_pp = pp.clone()

    obs_1 = torch.randn(P, 8, 6)
    obs_2 = torch.randn(P, 8, 6)
    true_actions = torch.randint(0, 3, (P, 8))

    # 2 differentiable inner Adam steps using meta_net's learned targets
    cur_pp = pp
    for chunk_obs in (obs_1, obs_2):
        logits = pop(chunk_obs, params = cur_pp, per_individual = True)
        with torch.no_grad():
            ema_logits = pop(chunk_obs, params = ema_pp, per_individual = True)
            assert ema_logits.shape == (P, 8, 3)

        target_logits = meta_net(chunk_obs)
        inner_loss = F.kl_div(
            logits.log_softmax(dim = -1),
            target_logits.softmax(dim = -1),
            reduction = 'batchmean',
        )
        cur_pp = inner_opt.functional_step(inner_loss, cur_pp, create_graph = True)

    # Evaluate updated population and backprop meta-loss into meta_net
    post_logits = pop(obs_2, params = cur_pp, per_individual = True)
    meta_loss = F.cross_entropy(post_logits.reshape(-1, 3), true_actions.reshape(-1))

    meta_opt.zero_grad()
    meta_loss.backward()

    meta_grad_norm = sum(p.grad.abs().sum().item() for p in meta_net.parameters() if p.grad is not None)
    assert meta_grad_norm > 0.0
    assert torch.isfinite(torch.tensor(meta_grad_norm))
    meta_opt.step()

    # TBPTT truncation: detach entire PopulationParameters (params + opt state) in one call
    next_pp = cur_pp.detach()
    assert next_pp.flat_params.grad_fn is None
    assert next_pp.flat_params.requires_grad is True
    assert next_pp.flat_exp_avg.grad_fn is None

    # Sync detached state back onto pop, reset masked agents, sync masked EMA, and update EMA
    pop.population_parameters.copy_from_(next_pp)

    unmasked_before = pop.population_parameters.flat_params[[1, 3, 5]].detach().clone()
    mask = torch.tensor([True, False, True, False, True, False])
    pop.reset_(mask)
    ema_pp.copy_from_(pop.population_parameters, members = mask, include_opt = False)
    assert torch.equal(ema_pp.flat_params[mask], pop.population_parameters.flat_params[mask])
    ema_pp.lerp_(pop.population_parameters, 0.05)

    assert pop.template is not None
    assert torch.equal(pop.population_parameters.flat_params[[1, 3, 5]], unmasked_before)
    assert not torch.allclose(pop.population_parameters.flat_params[0], next_pp.flat_params[0])
    if not shared:
        assert torch.equal(pop.population_parameters.flat_exp_avg[mask], torch.zeros_like(pop.population_parameters.flat_exp_avg[mask]))
        assert torch.equal(pop.population_parameters.flat_steps[mask], torch.zeros_like(pop.population_parameters.flat_steps[mask]))
        assert (pop.population_parameters.flat_steps[~mask] == 2).all()


def test_functional_step_matches_inplace_adam_numerically():
    torch.manual_seed(7)
    nets = [nn.Sequential(nn.Linear(8, 16), nn.SiLU(), nn.Linear(16, 4)) for _ in range(3)]
    pop_inplace = Population(nets)
    pop_func = Population([deepcopy(m) for m in nets])

    opt_inplace = pop_inplace.adam(lr = 1e-2, shared = False)
    opt_func = pop_func.adam(lr = 1e-2, shared = False)

    pp_func = pop_func.population_parameters
    for _ in range(5):
        x_pop = torch.randn(3, 6, 8)
        y_pop = torch.randn(3, 6, 4)

        opt_inplace.zero_grad()
        (pop_inplace(x_pop, per_individual = True) - y_pop).pow(2).mean(dim = (1, 2)).sum().backward()
        opt_inplace.step()

        loss_f = (pop_func(x_pop, params = pp_func, per_individual = True) - y_pop).pow(2).mean(dim = (1, 2)).sum()
        pp_func = opt_func.functional_step(loss_f, pp_func, create_graph = False)

    assert torch.allclose(pop_inplace.population_parameters.flat_params, pp_func.flat_params, atol = 1e-6)
    assert torch.allclose(pop_inplace.population_parameters.flat_exp_avg, pp_func.flat_exp_avg, atol = 1e-6)
    assert torch.allclose(pop_inplace.population_parameters.flat_exp_avg_sq, pp_func.flat_exp_avg_sq, atol = 1e-6)


# 5. Colocated gradient lanes, per-lane EMAs, meta tensors, and population mean soup

def test_colocated_gradient_lanes_lane_emas_meta_and_mean():
    torch.manual_seed(11)
    net = RichNet()
    pop = Population(net, pop_size = 4)
    opt = pop.adam(lr = 1e-2, shared = False)
    pp = pop.population_parameters

    x = torch.randn(8, 8)
    y = torch.randn(8, 4)

    for _ in range(3):
        preds = pop(x)
        loss_task = (preds - y.unsqueeze(0)).pow(2).mean(dim = (1, 2)).sum()
        loss_reg = preds.pow(2).mean(dim = (1, 2)).sum() * 0.1

        # compute two independent flat gradient lanes directly on flat_params
        g_task, = torch.autograd.grad(loss_task, (pp.flat_params,), retain_graph = True)
        g_reg, = torch.autograd.grad(loss_reg, (pp.flat_params,))

        pp.set_lane('task', g_task)
        pp.set_lane('reg', g_reg)

        # per-lane gradient EMAs & parameter EMA & per-individual meta stats — all colocated!
        ema_task = pp.update_ema('task_ema', 'grad.task', decay = 0.8)
        ema_reg = pp.update_ema('reg_ema', 'grad.reg', decay = 0.8)
        pp.update_ema('weights', 'params', decay = 0.9)

        # per-individual cosine similarity between task and reg lanes as a colocated meta tensor
        cos = F.cosine_similarity(g_task, g_reg, dim = -1)
        pp.ema_meta('task_reg_cos', cos, decay = 0.5)

        # custom combination using lane + smoothed lane EMA (PCGrad-style conflict projection on flat (P, D)!)
        dot = (ema_reg * ema_task).sum(dim = -1, keepdim = True)
        denom = (ema_task * ema_task).sum(dim = -1, keepdim = True).clamp_min(1e-12)
        g_reg_proj = ema_reg - (dot.clamp_max(0.) / denom) * ema_task
        pp.set_lane('combined', g_task + g_reg_proj)

        opt.step(lane = 'combined')

    assert pp.lane('task').shape == pp.flat_params.shape
    assert pp.ema('task_ema').shape == pp.flat_params.shape
    assert pp.ema('weights').shape == pp.flat_params.shape
    assert pp.meta['task_reg_cos'].shape == (4,)
    assert pp.meta['ema.task_reg_cos'].shape == (4,)

    # individual_dict exposes colocated lanes, emas, and meta for individual 1
    d1 = pop.individual_dict(1)
    assert 'task' in d1['lanes'] and 'combined' in d1['lanes']
    assert 'task_ema' in d1['emas'] and 'weights' in d1['emas']
    assert 'task_reg_cos' in d1['meta']

    # masked reset resets params, opt state, params-EMA (to new params), grad-EMA (to 0), and meta (to 0)
    mask = torch.tensor([True, False, True, False])
    pop.reset_(mask)
    assert torch.equal(pp.ema('weights')[mask], pp.flat_params[mask])
    assert torch.equal(pp.ema('task_ema')[mask], torch.zeros_like(pp.ema('task_ema')[mask]))
    assert torch.equal(pp.meta['task_reg_cos'][mask], torch.zeros_like(pp.meta['task_reg_cos'][mask]))

    # pop.mean() returns a bare single RichNet with averaged population weights
    avg_net = pop.mean()
    assert isinstance(avg_net, RichNet)
    assert torch.allclose(avg_net.head.weight, pop.params_dict['head.weight'].mean(dim = 0), atol = 1e-6)

