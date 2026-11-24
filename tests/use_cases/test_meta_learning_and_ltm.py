import pytest
import torch
from torch import nn

from torch_population import Population


def make_double_mlp(dim_in = 8, dim_hidden = 16, dim_out = 4):
    return nn.Sequential(nn.Linear(dim_in, dim_hidden), nn.SiLU(), nn.Linear(dim_hidden, dim_out)).double()


@pytest.mark.parametrize('opt_cls, opt_kwargs', (
    (torch.optim.SGD, dict(lr = 1e-1, momentum = 0.9)),
    (torch.optim.SGD, dict(lr = 1e-1, momentum = 0.9, nesterov = True)),
    (torch.optim.Adam, dict(lr = 1e-2)),
    (torch.optim.AdamW, dict(lr = 1e-2, weight_decay = 1e-2)),
    (torch.optim.RMSprop, dict(lr = 1e-2, alpha = 0.9)),
    (torch.optim.Adagrad, dict(lr = 5e-2)),
))
def test_unrolled_backprop_finite_diff_parity_across_optimizers(opt_cls, opt_kwargs):
    torch.manual_seed(400)
    x_tr = torch.randn(3, 6, 8, dtype = torch.float64)
    x_val = torch.randn(3, 6, 8, dtype = torch.float64)

    pop = Population(make_double_mlp, pop_size = 3)
    opt = pop.optimizer(opt_cls, **opt_kwargs)
    meta_target = nn.Parameter(torch.randn(1, 1, 4, dtype = torch.float64))

    def unroll(target_val, *, create_graph = True):
        pp = pop.population_parameters
        for _ in range(3):
            loss = (pop(x_tr, params = pp, per_individual = True) - target_val).pow(2).mean()
            pp = opt.functional_step(loss, pp, create_graph = create_graph, detach_preconditioner = False)
        return pop(x_val, params = pp, per_individual = True).pow(2).mean()

    unroll(meta_target, create_graph = True).backward()
    g_meta = meta_target.grad.clone()
    g_init = pop.population_parameters._pop_param.grad.clone()
    assert g_meta.isfinite().all() and g_meta.abs().sum() > 0
    assert g_init.isfinite().all() and g_init.abs().sum() > 0

    eps = 1e-6
    t_plus, t_minus = meta_target.detach().clone(), meta_target.detach().clone()
    t_plus[0, 0, 0] += eps
    t_minus[0, 0, 0] -= eps
    fd = (unroll(t_plus, create_graph = False) - unroll(t_minus, create_graph = False)) / (2 * eps)
    rel_err = (g_meta[0, 0, 0] - fd).abs() / fd.abs().clamp(min = 1e-12)
    assert rel_err < 1e-4, f'{opt_cls.__name__} {opt_kwargs} FD mismatch: {g_meta[0,0,0]} vs {fd}'


def test_mixed_optimizers_and_ltm_consolidation_edge_cases():
    torch.manual_seed(401)
    x_tr = torch.randn(4, 6, 8, dtype = torch.float64)
    x_val = torch.randn(4, 6, 8, dtype = torch.float64)

    # include a shared bias to verify shared layers stay untouched by single-individual LTM consolidation
    pop = Population(make_double_mlp, pop_size = 4, shared = ('0.bias',))
    ltm = pop.clone(include_opt = False)
    ltm_opt = ltm.optimizer(torch.optim.AdamW, lr = 1e-1)
    ltm_solo = Population(pop[0])

    opt_sgd   = pop.optimizer(torch.optim.SGD, lr = 5e-2, momentum = 0.9, name = 'sgd')
    opt_rms   = pop.optimizer(torch.optim.RMSprop, lr = 1e-2, name = 'rms')
    opt_adamw = pop.optimizer(torch.optim.AdamW, lr = 1e-2, name = 'adamw')
    meta_target = nn.Parameter(torch.randn(1, 1, 4, dtype = torch.float64))

    # 1. Alternating SGD(momentum) -> RMSprop -> AdamW with exact finite-diff check
    def unroll_mixed(target_val, *, create_graph = True):
        fast = pop.population_parameters
        for step_opt in (opt_sgd, opt_rms, opt_adamw):
            loss = (pop(x_tr, params = fast, per_individual = True) - target_val).pow(2).mean()
            fast = step_opt.functional_step(loss, fast, create_graph = create_graph, detach_preconditioner = False)
        return pop(x_val, params = fast, per_individual = True).pow(2).mean(), fast

    meta_loss, fast = unroll_mixed(meta_target, create_graph = True)
    meta_loss.backward()
    g_mixed = meta_target.grad.clone()

    t_plus, t_minus = meta_target.detach().clone(), meta_target.detach().clone()
    t_plus[0, 0, 0] += 1e-6
    t_minus[0, 0, 0] -= 1e-6
    fd = (unroll_mixed(t_plus, create_graph = False)[0] - unroll_mixed(t_minus, create_graph = False)[0]) / 2e-6
    assert ((g_mixed[0, 0, 0] - fd).abs() / fd.abs().clamp(min = 1e-12)) < 1e-4

    # 2. All 5 LTM consolidation styles + edge cases
    shared_before = ltm.params_dict['0.bias'].detach().clone()
    w0_before = ltm.view(0).flat_params.detach().clone()
    w1_before = ltm.view(1).flat_params.detach().clone()

    # 2a. individual view lerp_ from functional fast_pp (shared bias & other members untouched)
    ltm.view(0).lerp_(fast, 0.1)
    assert torch.allclose(ltm.view(0).flat_params, w0_before.lerp(fast.flat_params[0].detach(), 0.1), atol = 1e-12)
    assert torch.equal(ltm.view(1).flat_params, w1_before)
    assert torch.equal(ltm.params_dict['0.bias'], shared_before)

    # 2b. cross-member view lerp_ (member 2 -> member 0's LTM) & multi-view lerp_
    w0_mid = ltm.view(0).flat_params.detach().clone()
    ltm.view(0).lerp_(pop.view(2), 0.2)
    assert torch.allclose(ltm.view(0).flat_params, w0_mid.lerp(pop.view(2).flat_params.detach(), 0.2), atol = 1e-12)

    w13_before = ltm.view([1, 3]).flat_params.detach().clone()
    ltm.view([1, 3]).lerp_(pop.view([2, 0]), torch.tensor([0.25, 0.75]))
    assert torch.allclose(ltm.view(1).flat_params, w13_before[0].lerp(pop.view(2).flat_params.detach(), 0.25), atol = 1e-12)
    assert torch.allclose(ltm.view(3).flat_params, w13_before[1].lerp(pop.view(0).flat_params.detach(), 0.75), atol = 1e-12)

    # mismatched view sizes raise clear ValueError
    with pytest.raises(ValueError):
        ltm.view([0, 1]).lerp_(pop.view([0, 1, 2]), 0.1)

    # 2c. standalone single-module LTM from functional fast[0] dict AND from raw nn.Module
    w0_solo = ltm_solo.flat_params.detach().clone()
    ltm_solo.lerp_(fast[0], 0.1)
    assert not torch.allclose(ltm_solo.flat_params, w0_solo)
    raw_mod = pop[2]
    w0_solo2 = ltm_solo.flat_params.detach().clone()
    ltm_solo.lerp_(raw_mod, 0.5)
    assert not torch.allclose(ltm_solo.flat_params, w0_solo2)

    # 2d. Reptile / outer-optimizer LTM update on a single member
    w2_before = ltm.view(2).flat_params.detach().clone()
    w1_chk = ltm.view(1).flat_params.detach().clone()
    ltm_opt.step(grads = ltm.flat_params - fast.flat_params.detach(), members = 2)
    assert not torch.allclose(ltm.view(2).flat_params, w2_before)
    assert torch.equal(ltm.view(1).flat_params, w1_chk)

    # 2e. colocated per-individual LTM via view.ema_ -> runnable via params=view.tracked('ltm')
    pop.view(0).ema_('ltm', fast.flat_params[0], decay = 0.95)
    assert pop(x_val[0], params = pop.view(0).tracked('ltm')).shape == (6, 4)
    # reset individual 0 -> its colocated LTM resets & re-inits fresh without zero-bias
    pop.view(0).reset_()
    assert pop.view(0).tracked('ltm').abs().sum() == 0
    pop.view(0).ema_('ltm', fast.flat_params[0], decay = 0.95)
    assert torch.allclose(pop.view(0).tracked('ltm'), fast.flat_params[0].detach())

    # 3. Heterogeneous optimizers per member in one inner loop (0,1=AdamW; 2=RMSprop; 3=untouched!)
    meta_target.grad = None
    pop.zero_grad()
    fast_h = pop.population_parameters
    w3_init = fast_h.flat_params[3].detach().clone()
    for _ in range(2):
        l_a = (pop(x_tr, params = fast_h, per_individual = True) - meta_target).pow(2).mean()
        fast_h = opt_adamw.functional_step(l_a, fast_h, members = (0, 1), create_graph = True)
        l_b = (pop(x_tr, params = fast_h, per_individual = True) - meta_target).pow(2).mean()
        fast_h = opt_rms.functional_step(l_b, fast_h, members = 2, create_graph = True)

    assert torch.equal(fast_h.flat_params[3], w3_init)  # member 3 untouched!
    pop(x_val, params = fast_h, per_individual = True).pow(2).mean().backward()
    assert meta_target.grad.isfinite().all() and meta_target.grad.abs().sum() > 0
    assert (pop.population_parameters._pop_param.grad[:3].abs().sum(dim = -1) > 0).all()
