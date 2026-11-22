from copy import deepcopy
import pytest
import torch
from torch import nn

from torch_population import Adam, AdamW, Population


def make_mlp(seed: int = 0) -> nn.Sequential:
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(8, 16),
        nn.SiLU(),
        nn.Linear(16, 4),
    )


# 1. Shared parameters across population + subset forward


def test_shared_parameters_and_subset_forward():
    models = [make_mlp(i) for i in range(4)]
    pop = Population(models, shared = ('0.bias',))
    opt = pop.adamw(lr = 1e-2)
    pp = pop.population_parameters

    assert pop.pop_size == 4
    assert pp.is_shared_param('0.bias')
    assert not pp.is_shared_param('0.weight')
    assert pop.params_dict['0.bias'].shape == (16,)
    assert pop.params_dict['0.weight'].shape == (4, 16, 8)
    assert pp.flat_shared_params is not None
    assert pp.flat_shared_params.shape == (16,)

    # Shared param initialized as mean of source modules
    expected_bias = torch.stack([m[0].bias.detach() for m in models]).mean(0)
    assert torch.allclose(pop.params_dict['0.bias'], expected_bias)

    x = torch.randn(12, 8)
    y = torch.randn(12, 4)

    # Full forward & step updates both populated and shared params
    preds = pop(x)
    assert preds.shape == (4, 12, 4)
    w_before = pop.params_dict['0.weight'].detach().clone()
    b_before = pop.params_dict['0.bias'].detach().clone()

    opt.zero_grad()
    (preds - y.unsqueeze(0)).pow(2).mean().backward()
    opt.step()

    assert not torch.allclose(pop.params_dict['0.weight'], w_before)
    assert not torch.allclose(pop.params_dict['0.bias'], b_before)

    # Subset forward with tuple & bool mask
    sub_preds = pop(x, members = (0, 2))
    assert sub_preds.shape == (2, 12, 4)
    assert torch.allclose(sub_preds[0], pop[0](x), atol = 1e-5)
    assert torch.allclose(sub_preds[1], pop[2](x), atol = 1e-5)

    mask = torch.tensor([False, True, False, True])
    mask_preds = pop(x, members = mask)
    assert mask_preds.shape == (2, 12, 4)
    assert torch.allclose(mask_preds[0], pop[1](x), atol = 1e-5)

    # Subset Adam step only updates selected members on populated params
    w_before2 = pop.params_dict['0.weight'].detach().clone()
    opt.zero_grad()
    (pop(x, members = (0, 2)) - y.unsqueeze(0)).pow(2).mean().backward()
    opt.step(members = (0, 2))

    assert not torch.allclose(pop.params_dict['0.weight'][0], w_before2[0])
    assert not torch.allclose(pop.params_dict['0.weight'][2], w_before2[2])
    assert torch.allclose(pop.params_dict['0.weight'][1], w_before2[1])
    assert torch.allclose(pop.params_dict['0.weight'][3], w_before2[3])

    # Extraction and single-individual dict work with shared params
    ind, ind_opt = pop.extract(1)
    assert torch.allclose(ind[0].bias, pop.params_dict['0.bias'])
    assert torch.allclose(ind[0].weight, pop.params_dict['0.weight'][1])
    d1 = pop.individual_dict(1)
    assert d1['named_parameters']['0.bias'].shape == (16,)
    assert d1['named_parameters']['0.weight'].shape == (16, 8)
    assert '0.bias' in d1['named_optimizer_state']


def test_single_module_populate_with_shared():
    net = make_mlp(0)
    pop = Population(net, 3, init = 'clone', shared = ('0.bias',))
    assert pop.pop_size == 3
    assert pop.params_dict['0.bias'].shape == (16,)
    assert pop.params_dict['0.weight'].shape == (3, 16, 8)
    assert torch.allclose(pop.params_dict['0.weight'][0], pop.params_dict['0.weight'][2])


# 2. Population lifecycle: replace, delete, keep, append resize ALL colocated tensors in lockstep


def test_population_lifecycle_replace_delete_keep_append():
    models = [make_mlp(i) for i in range(4)]
    pop = Population(models, shared = ('0.bias',))
    opt = pop.adamw(lr = [1e-2, 2e-2, 3e-2, 4e-2])
    pp = pop.population_parameters

    x = torch.randn(10, 8)
    y = torch.randn(10, 4)

    for _ in range(3):
        opt.zero_grad()
        preds = pop(x)
        losses = (preds - y.unsqueeze(0)).pow(2).mean(dim = (1, 2))
        pp.ema_meta('loss', losses, decay = 0.8)
        losses.mean().backward()
        opt.step()
        pp.update_ema('weights', 'params', decay = 0.9)

    # Also attach a gradient lane and per-lane EMA
    g_task = torch.randn_like(pp.flat_params)
    pp.set_lane('task', g_task)
    pp.update_lane_ema('task', decay = 0.8)
    pp.update_lane_var('task', decay = 0.9)

    assert pp.flat_params.shape[0] == 4
    assert pp.flat_exp_avg.shape[0] == 4
    assert pp.ema('weights').shape[0] == 4
    assert pp.meta['loss'].shape == (4,)
    assert pp.lane_meta('task')['ema'].shape[0] == 4
    assert pp.lane_meta('task')['var'].shape[0] == 4

    # Replace member 3 with member 0, resetting opt/ema/meta/lanes
    pop.replace(3, 0, reset_opt = True, reset_ema = True, reset_meta = True)
    assert torch.allclose(pop.params_dict['0.weight'][3], pop.params_dict['0.weight'][0])
    assert pp.flat_exp_avg[3].norm() == 0
    assert pp.flat_exp_avg[0].norm() > 0
    assert pp.ema('weights')[3].norm() == 0
    assert pp.meta['loss'][3].item() == 0.

    # Delete member 1 -> P: 4 -> 3 across params, opt, per-ind lr, ema, meta, lanes, lane_meta
    w2_before = pop.params_dict['0.weight'][2].detach().clone()
    pop.delete(1)
    assert pop.pop_size == 3
    assert pp.flat_params.shape[0] == 3
    assert pp.flat_exp_avg.shape[0] == 3
    assert pp.flat_steps.shape[0] == 3
    assert pp.ema('weights').shape[0] == 3
    assert pp.meta['loss'].shape == (3,)
    assert pp.lane('task').shape[0] == 3
    assert pp.lane_meta('task')['ema'].shape[0] == 3
    assert pp.lane_meta('task')['var'].shape[0] == 3
    assert torch.allclose(pop.params_dict['0.weight'][1], w2_before)
    assert torch.allclose(opt.param_groups[0]['lr'], torch.tensor([1e-2, 3e-2, 4e-2]))

    # Keep members (0, 2) -> P: 3 -> 2
    pop.keep((0, 2))
    assert pop.pop_size == 2
    assert pp.flat_params.shape[0] == 2
    assert pp.flat_exp_avg.shape[0] == 2
    assert pp.ema('weights').shape[0] == 2
    assert pp.meta['loss'].shape == (2,)
    assert pp.lane_meta('task')['ema'].shape[0] == 2

    # Append a fresh module and clone of member 0 -> P: 2 -> 3 -> 4
    fresh = make_mlp(99)
    idx_a = pop.append(fresh)
    idx_b = pop.append(0)
    assert idx_a == 2 and idx_b == 3
    assert pop.pop_size == 4
    assert torch.allclose(pop.params_dict['0.weight'][2], fresh[0].weight)
    assert torch.allclose(pop.params_dict['0.weight'][3], pop.params_dict['0.weight'][0])
    assert pp.flat_exp_avg[2].norm() == 0
    assert pp.flat_exp_avg[3].norm() == 0
    assert pp.lane_meta('task')['ema'][2].norm() == 0

    # Training continues cleanly with attached Adam after delete + keep + append
    opt.zero_grad()
    preds = pop(x)
    assert preds.shape == (4, 10, 4)
    (preds - y.unsqueeze(0)).pow(2).mean().backward()
    opt.step()
    assert pp.flat_exp_avg[2].norm() > 0


# 3. Generic colocated _FlatOptStore on PopulationParameters


def test_generic_flat_opt_store_colocated_and_resizes():
    models = [make_mlp(i) for i in range(3)]
    pop = Population(models, shared = ('0.bias',))
    pp = pop.population_parameters

    # Register a Muon-style split optimizer store (matrix params -> 'm', vector params -> 'm1', 'm2')
    store = pp.register_opt_store('muon')
    mat_names = [s.name for s in pp.trainable_specs if len(s.shape) >= 2]
    vec_names = [s.name for s in pp.trainable_specs if len(s.shape) < 2]
    store.alloc_slot('m', mat_names)
    store.alloc_slot('m1', vec_names)
    store.alloc_slot('m2', vec_names)
    store.alloc_global('t', torch.zeros((), dtype = torch.float32))

    # Zero-copy views into colocated flat buffers
    m_views = store.slot_views('m')
    assert m_views['0.weight'].shape == (3, 16, 8)
    m1_views = store.slot_views('m1')
    assert m1_views['0.bias'].shape == (16,)      # shared param -> unpopulated!
    assert m1_views['2.bias'].shape == (3, 4)     # populated param -> (P, 4)!

    with torch.no_grad():
        m_views['0.weight'][1].fill_(7.0)
        m1_views['2.bias'][2].fill_(3.0)
        store.global_tensor('t').fill_(5.0)

    assert torch.allclose(store.slot_views('m')['0.weight'][1], torch.full((16, 8), 7.0))

    # Resize population via delete & append -> store resizes in lockstep!
    pop.delete(0)
    assert store.slot_views('m')['0.weight'].shape == (2, 16, 8)
    assert torch.allclose(store.slot_views('m')['0.weight'][0], torch.full((16, 8), 7.0))
    assert store.slot_views('m1')['0.bias'].shape == (16,)
    assert torch.allclose(store.slot_views('m1')['2.bias'][1], torch.full((4,), 3.0))

    pop.append(0)
    assert store.slot_views('m')['0.weight'].shape == (3, 16, 8)
    assert store.slot_views('m')['0.weight'][2].norm() == 0

    # Reset single member zeroes only that member's populated slots
    store.reset_members_([0])
    assert store.slot_views('m')['0.weight'][0].norm() == 0
    assert torch.allclose(store.slot_views('m1')['2.bias'][1], torch.full((4,), 3.0))
