from copy import deepcopy

import pytest
import torch
from torch import nn
from torch.nn import Module

from torch_population import Adam, AdamW, ParamSlot, Population

param = pytest.mark.parametrize


def _make_mlp():
    return nn.Sequential(nn.Linear(8, 16), nn.SiLU(), nn.Linear(16, 4))


# 1. Exact numerical equivalence with torch.optim.Adam and torch.optim.AdamW

@param('use_adamw', [False, True])
def test_noop_matches_torch_adam_exactly(use_adamw):
    torch.manual_seed(0)
    net_a = _make_mlp()
    net_b = deepcopy(net_a)

    pop = Population(net_a)
    if use_adamw:
        opt_a = pop.adamw(lr = 1e-2, betas = (0.9, 0.99), eps = 1e-8, weight_decay = 1e-2)
        opt_b = torch.optim.AdamW(net_b.parameters(), lr = 1e-2, betas = (0.9, 0.99), eps = 1e-8, weight_decay = 1e-2)
    else:
        opt_a = pop.adam(lr = 1e-2, betas = (0.9, 0.99), eps = 1e-8, weight_decay = 1e-2)
        opt_b = torch.optim.Adam(net_b.parameters(), lr = 1e-2, betas = (0.9, 0.99), eps = 1e-8, weight_decay = 1e-2)

    for _ in range(10):
        x = torch.randn(6, 8)
        y = torch.randn(6, 4)

        opt_a.zero_grad()
        (pop(x) - y).pow(2).mean().backward()
        opt_a.step()

        opt_b.zero_grad()
        (net_b(x) - y).pow(2).mean().backward()
        opt_b.step()

    for pa, pb in zip(pop.parameters(), net_b.parameters()):
        assert torch.allclose(pa, pb, atol = 1e-6)


@param('use_adamw', [False, True])
def test_population_separate_matches_independent_torch_adams(use_adamw):
    torch.manual_seed(1)
    nets = [_make_mlp() for _ in range(4)]
    refs = [deepcopy(m) for m in nets]

    pop = Population(nets)
    if use_adamw:
        opt = AdamW(pop, lr = 1e-2, betas = (0.9, 0.999), eps = 1e-8, weight_decay = 2e-2, shared = False)
        ref_opts = [torch.optim.AdamW(m.parameters(), lr = 1e-2, betas = (0.9, 0.999), eps = 1e-8, weight_decay = 2e-2) for m in refs]
    else:
        opt = Adam(pop, lr = 1e-2, betas = (0.9, 0.999), eps = 1e-8, weight_decay = 0., shared = False)
        ref_opts = [torch.optim.Adam(m.parameters(), lr = 1e-2, betas = (0.9, 0.999), eps = 1e-8, weight_decay = 0.) for m in refs]

    assert not opt.is_shared
    assert opt.shared_slots == frozenset()

    for _ in range(12):
        x_pop = torch.randn(4, 6, 8)
        y_pop = torch.randn(4, 6, 4)

        opt.zero_grad()
        preds = pop(x_pop, per_individual = True)
        losses = (preds - y_pop).pow(2).mean(dim = (1, 2))
        losses.sum().backward()
        opt.step()

        for i, (m, ro) in enumerate(zip(refs, ref_opts)):
            ro.zero_grad()
            (m(x_pop[i]) - y_pop[i]).pow(2).mean().backward()
            ro.step()

    for i, ref in enumerate(refs):
        ind = pop[i]
        for pa, pb in zip(ind.parameters(), ref.parameters()):
            assert torch.allclose(pa, pb, atol = 1e-6)


# 2. Colocation, state_dict, slot inspection, and dtype movement

def test_colocated_storage_slot_state_dict_and_dtype():
    torch.manual_seed(2)
    net = _make_mlp()
    pop = Population(net, pop_size = 3)
    num_params_before = sum(p.numel() for p in pop.parameters())

    opt = Adam(pop, lr = 1e-2, shared = False)

    # optimizer states are colocated buffers, NOT extra nn.Parameters
    assert sum(p.numel() for p in pop.parameters()) == num_params_before
    assert pop.opt is opt

    slot = pop.slot('0.weight')
    assert isinstance(slot, ParamSlot)
    assert slot.param.shape == (3, 16, 8)
    assert slot.exp_avg.shape == (3, 16, 8)
    assert slot.exp_avg_sq.shape == (3, 16, 8)
    assert slot.step.shape == (3,)
    assert slot.step.dtype == torch.int64

    x = torch.randn(5, 8)
    for _ in range(3):
        opt.zero_grad()
        pop(x).pow(2).mean().backward()
        opt.step()

    assert torch.equal(pop.slot('0.weight').step, torch.tensor([3, 3, 3]))

    # state_dict saves and restores BOTH weights AND colocated optimizer states
    pop2 = Population(_make_mlp(), pop_size = 3)
    opt2 = Adam(pop2, lr = 1e-2, shared = False)
    pop2.load_state_dict(pop.state_dict())

    for name in pop.params_dict:
        assert torch.equal(pop2.params_dict[name], pop.params_dict[name])
        assert torch.equal(opt2.exp_avg[name], opt.exp_avg[name])
        assert torch.equal(opt2.exp_avg_sq[name], opt.exp_avg_sq[name])
        assert torch.equal(opt2.steps[name], opt.steps[name])

    # dtype conversion moves params and colocated float moments while keeping int64 steps
    pop.double()
    slot64 = pop.slot('0.weight')
    assert slot64.param.dtype == torch.float64
    assert slot64.exp_avg.dtype == torch.float64
    assert slot64.exp_avg_sq.dtype == torch.float64
    assert slot64.step.dtype == torch.int64

    opt.zero_grad()
    pop(x.double()).pow(2).mean().backward()
    opt.step()
    assert torch.equal(pop.slot('0.weight').step, torch.tensor([4, 4, 4]))


# 3. Shared vs separate vs granular sharing, plus runtime share() / unshare()

def test_shared_vs_separate_vs_granular_shapes_and_training():
    torch.manual_seed(3)
    x = torch.randn(32, 8)
    target = torch.sin(x @ torch.randn(8, 4))

    # Fully shared (1/P optimizer memory)
    pop_shared = Population(_make_mlp(), pop_size = 4, init = 'clone', noise_std = 0.02)
    opt_shared = pop_shared.adam(lr = 1e-2, shared = True)

    assert opt_shared.is_shared
    assert opt_shared.exp_avg['0.weight'].shape == (16, 8)
    assert opt_shared.exp_avg_sq['0.weight'].shape == (16, 8)
    assert opt_shared.steps['0.weight'].shape == ()

    init_loss = (pop_shared(x) - target.unsqueeze(0)).pow(2).mean().item()
    for _ in range(150):
        opt_shared.zero_grad()
        (pop_shared(x) - target.unsqueeze(0)).pow(2).mean(dim = (1, 2)).sum().backward()
        opt_shared.step()
    final_loss = (pop_shared(x) - target.unsqueeze(0)).pow(2).mean().item()
    assert final_loss < init_loss * 0.15

    # Granular sharing: share second moment `v` only, keep `m` and `step` separate
    pop_v = Population(_make_mlp(), pop_size = 4, init = 'reinit')
    opt_v = pop_v.adam(lr = 1e-2, shared = ('v',))
    assert not opt_v.is_shared
    assert opt_v.shared_slots == frozenset({'v'})
    assert opt_v.exp_avg['0.weight'].shape == (4, 16, 8)
    assert opt_v.exp_avg_sq['0.weight'].shape == (16, 8)
    assert opt_v.steps['0.weight'].shape == (4,)

    init_loss_v = (pop_v(x) - target.unsqueeze(0)).pow(2).mean().item()
    for _ in range(150):
        opt_v.zero_grad()
        (pop_v(x) - target.unsqueeze(0)).pow(2).mean(dim = (1, 2)).sum().backward()
        opt_v.step()
    final_loss_v = (pop_v(x) - target.unsqueeze(0)).pow(2).mean().item()
    assert final_loss_v < init_loss_v * 0.15

    # Runtime conversion: separate -> shared -> separate
    pop_rt = Population(_make_mlp(), pop_size = 3)
    opt_rt = Adam(pop_rt, lr = 1e-2, shared = False)
    opt_rt.zero_grad()
    pop_rt(x).pow(2).mean().backward()
    opt_rt.step()

    m_before = opt_rt.exp_avg['0.weight'].clone()
    opt_rt.share()
    assert opt_rt.is_shared
    assert opt_rt.exp_avg['0.weight'].shape == (16, 8)
    assert torch.allclose(opt_rt.exp_avg['0.weight'], m_before.mean(dim = 0))
    assert opt_rt.steps['0.weight'].shape == ()

    opt_rt.unshare()
    assert not opt_rt.is_shared
    assert opt_rt.exp_avg['0.weight'].shape == (3, 16, 8)
    assert torch.allclose(opt_rt.exp_avg['0.weight'][0], opt_rt.exp_avg['0.weight'][2])
    assert opt_rt.steps['0.weight'].shape == (3,)


# 4. Extract individual + optimizer, train solo, index back into population

@param('use_adamw', [False, True])
def test_extract_train_solo_and_index_back_exact_continuity(use_adamw):
    """Steps 1..5 in Population -> extract(2) -> steps 6..10 solo -> pop[2] = (ind, ind_opt) -> steps 11..15 in Population
    must match a single continuous torch.optim.Adam / AdamW trajectory to < 1e-6."""
    torch.manual_seed(4)
    nets = [_make_mlp() for _ in range(4)]
    ref_net2 = deepcopy(nets[2])

    pop = Population(nets)
    if use_adamw:
        opt = AdamW(pop, lr = 1e-2, betas = (0.9, 0.999), eps = 1e-8, weight_decay = 1e-2, shared = False)
        ref_opt2 = torch.optim.AdamW(ref_net2.parameters(), lr = 1e-2, betas = (0.9, 0.999), eps = 1e-8, weight_decay = 1e-2)
    else:
        opt = Adam(pop, lr = 1e-2, betas = (0.9, 0.999), eps = 1e-8, weight_decay = 0., shared = False)
        ref_opt2 = torch.optim.Adam(ref_net2.parameters(), lr = 1e-2, betas = (0.9, 0.999), eps = 1e-8, weight_decay = 0.)

    # Phase 1: steps 1..5 in Population
    for _ in range(5):
        x_pop = torch.randn(4, 6, 8)
        y_pop = torch.randn(4, 6, 4)

        opt.zero_grad()
        losses = (pop(x_pop, per_individual = True) - y_pop).pow(2).mean(dim = (1, 2))
        losses.sum().backward()
        opt.step()

        ref_opt2.zero_grad()
        (ref_net2(x_pop[2]) - y_pop[2]).pow(2).mean().backward()
        ref_opt2.step()

    # Snapshot other individuals (0, 1, 3) to verify they are completely untouched
    other_w_before = pop.params_dict['0.weight'][[0, 1, 3]].detach().clone()
    other_m_before = opt.exp_avg['0.weight'][[0, 1, 3]].detach().clone()
    other_step_before = opt.steps['0.weight'][[0, 1, 3]].detach().clone()

    # Phase 2: extract individual 2 + native torch.optim optimizer, train solo for steps 6..10
    ind, ind_opt = pop.extract(2)
    assert isinstance(ind, nn.Sequential)
    expected_cls = torch.optim.AdamW if use_adamw else torch.optim.Adam
    assert isinstance(ind_opt, expected_cls)

    for _ in range(5):
        x_solo = torch.randn(6, 8)
        y_solo = torch.randn(6, 4)

        ind_opt.zero_grad()
        (ind(x_solo) - y_solo).pow(2).mean().backward()
        ind_opt.step()

        ref_opt2.zero_grad()
        (ref_net2(x_solo) - y_solo).pow(2).mean().backward()
        ref_opt2.step()

    # Phase 3: index both updated individual & optimizer back into slot 2
    pop[2] = (ind, ind_opt)

    assert torch.equal(pop.params_dict['0.weight'][[0, 1, 3]], other_w_before)
    assert torch.equal(opt.exp_avg['0.weight'][[0, 1, 3]], other_m_before)
    assert torch.equal(opt.steps['0.weight'][[0, 1, 3]], other_step_before)
    assert int(opt.steps['0.weight'][2].item()) == 10

    # Phase 4: steps 11..15 back inside Population (notice member 2 has step=10 while others have step=5!)
    for _ in range(5):
        x_pop = torch.randn(4, 6, 8)
        y_pop = torch.randn(4, 6, 4)

        opt.zero_grad()
        losses = (pop(x_pop, per_individual = True) - y_pop).pow(2).mean(dim = (1, 2))
        losses.sum().backward()
        opt.step()

        ref_opt2.zero_grad()
        (ref_net2(x_pop[2]) - y_pop[2]).pow(2).mean().backward()
        ref_opt2.step()

    assert torch.equal(opt.steps['0.weight'], torch.tensor([10, 10, 15, 10]))

    final_ind2 = pop[2]
    for pa, pb in zip(final_ind2.parameters(), ref_net2.parameters()):
        assert torch.allclose(pa, pb, atol = 1e-6)


def test_checkout_context_manager_and_shared_extract_writeback():
    torch.manual_seed(5)
    pop = Population(_make_mlp(), pop_size = 3)
    opt = pop.adam(lr = 1e-2, shared = False)

    x = torch.randn(5, 8)
    with pop.checkout(1) as (ind, ind_opt):
        for _ in range(4):
            ind_opt.zero_grad()
            ind(x).pow(2).mean().backward()
            ind_opt.step()

    assert torch.equal(opt.steps['0.weight'], torch.tensor([0, 4, 0]))
    assert torch.allclose(pop(x)[1], ind(x), atol = 1e-6)

    # Also verify extract & writeback in shared=True mode
    pop_s = Population(_make_mlp(), pop_size = 3)
    opt_s = pop_s.adam(lr = 1e-2, shared = True)
    ind_s, ind_opt_s = opt_s.extract(0)
    for _ in range(3):
        ind_opt_s.zero_grad()
        ind_s(x).pow(2).mean().backward()
        ind_opt_s.step()

    pop_s[0] = (ind_s, ind_opt_s)
    assert int(opt_s.steps['0.weight'].item()) == 3
    assert torch.allclose(pop_s(x)[0], ind_s(x), atol = 1e-6)


def test_insert_fresh_unstepped_torch_optimizer_zeroes_slot():
    pop = Population(_make_mlp(), pop_size = 3)
    opt = pop.adam(lr = 1e-2, shared = False)

    x = torch.randn(5, 8)
    opt.zero_grad()
    pop(x).pow(2).mean().backward()
    opt.step()
    assert torch.equal(opt.steps['0.weight'], torch.tensor([1, 1, 1]))

    fresh_net = _make_mlp()
    fresh_opt = torch.optim.Adam(fresh_net.parameters(), lr = 1e-2)
    pop[1] = (fresh_net, fresh_opt)

    assert torch.equal(opt.steps['0.weight'], torch.tensor([1, 0, 1]))
    assert torch.equal(opt.exp_avg['0.weight'][1], torch.zeros_like(opt.exp_avg['0.weight'][1]))


# 5. Lifecycle: populate(), slice/resample, subset step, disco-rl reset, per-individual lrs, lr_scheduler, external grads

def test_noop_with_adam_then_populate_adapts_colocated_state():
    torch.manual_seed(6)
    net = _make_mlp()
    pop = Population(net)
    opt = pop.adam(lr = 1e-2, shared = False)

    x = torch.randn(5, 8)
    for _ in range(3):
        opt.zero_grad()
        pop(x).pow(2).mean().backward()
        opt.step()

    m0_before = opt.exp_avg['0.weight'].clone()
    pop.populate(4, init = 'reinit')

    assert opt.exp_avg['0.weight'].shape == (4, 16, 8)
    assert torch.equal(opt.exp_avg['0.weight'][0], m0_before)
    assert torch.equal(opt.exp_avg['0.weight'][1], torch.zeros_like(m0_before))
    assert torch.equal(opt.steps['0.weight'], torch.tensor([3, 0, 0, 0]))

    # stepping continues cleanly on the expanded population
    opt.zero_grad()
    pop(x).pow(2).mean().backward()
    opt.step()
    assert torch.equal(opt.steps['0.weight'], torch.tensor([4, 1, 1, 1]))


def test_slice_resample_and_slice_writeback_carry_colocated_opt():
    torch.manual_seed(7)
    pop = Population(_make_mlp(), pop_size = 4)
    opt = pop.adam(lr = 1e-2, shared = False)

    x = torch.randn(5, 8)
    opt.zero_grad()
    pop(x).pow(2).mean().backward()
    opt.step()

    sub = pop[[3, 0, 3]]
    assert sub.opt is not None
    assert torch.equal(sub.opt.exp_avg['0.weight'][0], opt.exp_avg['0.weight'][3])
    assert torch.equal(sub.opt.exp_avg['0.weight'][1], opt.exp_avg['0.weight'][0])
    assert torch.equal(sub.opt.exp_avg['0.weight'][2], opt.exp_avg['0.weight'][3])

    # train sub for 2 more steps and assign slice back into pop[0:2]
    for _ in range(2):
        sub.opt.zero_grad()
        sub(x).pow(2).mean().backward()
        sub.opt.step()

    pop[0:2] = sub[0:2]
    assert torch.equal(opt.steps['0.weight'], torch.tensor([3, 3, 1, 1]))
    assert torch.equal(opt.exp_avg['0.weight'][0], sub.opt.exp_avg['0.weight'][0])


def test_subset_step_reset_per_individual_lr_scheduler_and_external_grads():
    torch.manual_seed(8)
    lrs = [1e-4, 1e-3, 1e-2, 5e-2]
    pop = Population(_make_mlp(), pop_size = 4, init = 'clone')
    opt = pop.adam(lr = lrs, shared = False)

    x = torch.randn(6, 8)
    opt.zero_grad()
    pop.view([0, 2])(x).pow(2).mean().backward()
    opt.step(members = [0, 2])

    assert torch.equal(opt.steps['0.weight'], torch.tensor([1, 0, 1, 0]))
    assert torch.equal(opt.exp_avg['0.weight'][1], torch.zeros_like(opt.exp_avg['0.weight'][1]))

    # per-individual lr extracted properly
    _, ind_opt_3 = pop.extract(3)
    assert ind_opt_3.param_groups[0]['lr'] == pytest.approx(5e-2)

    # disco-rl style masked reset
    mask = torch.tensor([True, False, False, False])
    opt.reset(mask)
    assert torch.equal(opt.steps['0.weight'], torch.tensor([0, 0, 1, 0]))
    assert torch.equal(opt.exp_avg['0.weight'][0], torch.zeros_like(opt.exp_avg['0.weight'][0]))
    assert opt.exp_avg['0.weight'][2].abs().sum() > 0

    # external grads dict (grad-control interop) + LR scheduler interop
    pop_sched = Population(_make_mlp(), pop_size = 2)
    opt_sched = pop_sched.adam(lr = 1e-2, shared = False)
    sched = torch.optim.lr_scheduler.StepLR(opt_sched, step_size = 1, gamma = 0.5)

    custom_grads = {name: torch.ones_like(p) for name, p in pop_sched.params_dict.items()}
    opt_sched.step(grads = custom_grads)
    sched.step()
    assert opt_sched.param_groups[0]['lr'] == pytest.approx(5e-3)
    assert torch.equal(opt_sched.steps['0.weight'], torch.tensor([1, 1]))


def test_noop_extract_writeback_and_frozen_or_unused_params():
    torch.manual_seed(9)

    # noop mode extract -> solo train -> index back
    net = _make_mlp()
    pop = Population(net)
    opt = pop.adam(lr = 1e-2)

    x = torch.randn(5, 8)
    for _ in range(2):
        opt.zero_grad()
        pop(x).pow(2).mean().backward()
        opt.step()

    ind, ind_opt = pop.extract()
    assert ind is net
    for _ in range(3):
        ind_opt.zero_grad()
        ind(x).pow(2).mean().backward()
        ind_opt.step()
    pop[0] = (ind, ind_opt)
    assert int(opt.steps['0.weight'].item()) == 5

    # frozen parameters and unused parameters (grad is None)
    class TwoHead(Module):
        def __init__(self):
            super().__init__()
            self.head_a = nn.Linear(8, 4)
            self.head_b = nn.Linear(8, 4)
            self.head_a.bias.requires_grad_(False)

        def forward(self, x, use_b = False):
            return self.head_b(x) if use_b else self.head_a(x)

    pop_f = Population(TwoHead(), pop_size = 3)
    opt_f = pop_f.adam(lr = 1e-2, shared = False)

    assert 'head_a.bias' not in opt_f.exp_avg
    frozen_slot = pop_f.slot('head_a.bias')
    assert frozen_slot.exp_avg is None

    opt_f.zero_grad()
    pop_f(x, use_b = False).pow(2).mean().backward()
    opt_f.step()

    # head_a.weight stepped, head_b.weight had grad=None so its step stayed 0
    assert torch.equal(opt_f.steps['head_a.weight'], torch.tensor([1, 1, 1]))
    assert torch.equal(opt_f.steps['head_b.weight'], torch.tensor([0, 0, 0]))

    ind_f, ind_opt_f = pop_f.extract(1)
    assert ind_f.head_a.bias.requires_grad is False
    ind_opt_f.zero_grad()
    ind_f(x, use_b = True).pow(2).mean().backward()
    ind_opt_f.step()
    pop_f[1] = (ind_f, ind_opt_f)

    assert torch.equal(opt_f.steps['head_b.weight'], torch.tensor([0, 1, 0]))

