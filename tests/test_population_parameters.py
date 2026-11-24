from copy import deepcopy

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.func import functional_call
from torch.nn import Module

from torch_population import Population, PopulationParameters

param = pytest.mark.parametrize

# fixtures

def mlp(dim_in = 8, dim_hidden = 16, dim_out = 4, seed = None):
    if seed is not None:
        torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(dim_in, dim_hidden),
        nn.SiLU(),
        nn.Linear(dim_hidden, dim_out),
    )

class RichNet(Module):
    def __init__(self, dim_in = 8, dim_hidden = 16, dim_out = 4):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(dim_in, dim_hidden), nn.SiLU())
        self.head = nn.Linear(dim_hidden, dim_out)
        self.register_buffer('scale', torch.tensor(1.25))
        self.register_buffer('temp', torch.tensor(0.5), persistent = False)

    def forward(self, x):
        return self.head(self.encoder(x)) * self.scale * self.temp

def step_pop(pop, opt, x, y, *, per_individual = True):
    opt.zero_grad()
    target = y if per_individual else y.unsqueeze(0)
    (pop(x, per_individual = per_individual) - target).pow(2).mean(dim = (1, 2)).sum().backward()
    opt.step()

# tests

def test_physical_colocation_and_sharing():
    torch.manual_seed(0)
    net = RichNet()
    pop = Population(net, pop_size = 4)
    opt = pop.optimizer(torch.optim.Adam, lr = 1e-2, shared = False)
    pp = pop.population_parameters

    assert isinstance(pp, PopulationParameters)
    d = sum(p.numel() for p in net.parameters() if p.requires_grad)

    assert pp.flat_params.shape == (4, d)
    assert pp.flat_params.is_contiguous()
    for slot in ('exp_avg', 'exp_avg_sq'):
        assert pp.opt_slot_flat(slot).shape == (4, d)
        assert pp.opt_slot_flat(slot).is_contiguous()
    assert pp.opt_aux('step', 'head.weight').shape == (4,)

    # zero-copy views share exact underlying storage
    named_p = pp.named_parameters_dict()
    named_opt = pp.named_optimizer_state()
    for name, view in named_p.items():
        assert view.untyped_storage().data_ptr() == pp.flat_params.untyped_storage().data_ptr()
        for slot in ('exp_avg', 'exp_avg_sq'):
            assert named_opt[name][slot].untyped_storage().data_ptr() == pp.opt_slot_flat(slot).untyped_storage().data_ptr()

    # in-place flat mutation reflects immediately
    with torch.no_grad():
        pp.flat_params[1].zero_()
    assert torch.allclose(pop(torch.randn(5, 8))[1], torch.zeros(5, 4))
    assert torch.equal(pp[1]['named_parameters']['head.weight'], torch.zeros(4, 16))

    # share() collapses colocated state to 1/P
    opt.share()
    assert pp.opt_slot_flat('exp_avg').shape == (d,)
    assert pp.opt_aux('step', 'head.weight').shape == ()
    ptr = pp.opt_slot_flat('exp_avg').untyped_storage().data_ptr()
    assert pp[0]['named_optimizer_state']['head.weight']['exp_avg'].untyped_storage().data_ptr() == ptr
    assert pp[3]['named_optimizer_state']['head.weight']['exp_avg'].untyped_storage().data_ptr() == ptr

@param('opt_cls,kw', [
    (torch.optim.Adam, dict(lr = 1e-2, betas = (0.9, 0.999))),
    (torch.optim.AdamW, dict(lr = 1e-2, betas = (0.9, 0.999), weight_decay = 1e-2)),
    (torch.optim.RMSprop, dict(lr = 1e-2, momentum = 0.5)),
])
def test_individual_dict_pytorch_interop(opt_cls, kw):
    torch.manual_seed(1)
    net = RichNet()
    pop = Population(net, pop_size = 4)
    opt = pop.optimizer(opt_cls, **kw)

    for _ in range(5):
        step_pop(pop, opt, torch.randn(4, 6, 8), torch.randn(4, 6, 4))

    pp = pop.population_parameters
    d = pop.individual_dict(2)
    assert set(d) == set(pp[2])
    assert list(d['named_parameters']) == list(dict(net.named_parameters()))
    assert list(d['named_buffers']) == list(dict(net.named_buffers()))
    assert list(d['state_dict']) == list(net.state_dict())
    for t in d['named_parameters'].values():
        assert t.is_contiguous()
        assert t.untyped_storage().data_ptr() == pp.flat_params.untyped_storage().data_ptr()

    x = torch.randn(5, 8)
    fc_out = functional_call(deepcopy(net), (d['named_parameters'], d['named_buffers']), (x,))
    assert torch.allclose(fc_out, pop(x)[2], atol = 1e-6)

    # load into fresh native Module + native torch.optim and match step-for-step
    cloned = pop.individual_dict(2, clone = True)
    fresh = RichNet()
    fresh.load_state_dict(cloned['state_dict'])
    fresh_opt = opt_cls(fresh.parameters(), **kw)
    fresh_opt.load_state_dict(cloned['optimizer_state'])

    for _ in range(5):
        xp, yp = torch.randn(4, 6, 8), torch.randn(4, 6, 4)
        step_pop(pop, opt, xp, yp)
        fresh_opt.zero_grad()
        (fresh(xp[2]) - yp[2]).pow(2).mean().backward()
        fresh_opt.step()

    for name, p_ref in fresh.named_parameters():
        assert torch.allclose(pp[2]['named_parameters'][name], p_ref, atol = 1e-5)

    pp[0] = {
        'named_parameters': dict(fresh.named_parameters()),
        'named_buffers': dict(fresh.named_buffers()),
        'optimizer_state': fresh_opt.state_dict(),
    }
    assert torch.allclose(pop(x)[0], fresh(x), atol = 1e-6)
    assert int(pp[0]['named_optimizer_state']['head.weight']['step'].item()) == 10

    # sync_grads writes individual gradients back into colocated flat leaf grad
    pp.zero_grad_()
    fresh_opt.zero_grad()
    fresh(x).pow(2).mean().backward()
    pp.load_individual_dict_(1, {'named_parameters': dict(fresh.named_parameters())}, sync_grads = True)
    assert torch.allclose(pp.named_grads_dict()['head.weight'][1], fresh.head.weight.grad, atol = 1e-6)

def test_as_dict_and_noop():
    net = RichNet()
    pop_noop = Population(net)
    opt_noop = pop_noop.optimizer(torch.optim.RMSprop, lr = 1e-2)
    opt_noop.zero_grad()
    pop_noop(torch.randn(4, 8)).pow(2).mean().backward()
    opt_noop.step()

    d0 = pop_noop.individual_dict(0)
    assert d0['named_parameters']['head.weight'] is net.head.weight
    assert int(d0['named_optimizer_state']['head.weight']['step'].item()) == 1

    pop = Population(net, pop_size = 3)
    pop.optimizer(torch.optim.RMSprop, lr = 1e-2, shared = False)
    pd = pop.population_parameters.as_dict()
    assert pd['named_parameters']['head.weight'].shape == (3, 4, 16)
    assert pd['named_optimizer_state']['head.weight']['square_avg'].shape == (3, 4, 16)
    assert pd['named_optimizer_state']['head.weight']['step'].shape == (3,)
    assert len(pd['optimizer_state']['param_groups'][0]['params']) == len(pd['parameters'])

@param('shared', [False, True])
def test_meta_learning_functional_step(shared):
    torch.manual_seed(42)
    P = 6
    pop = Population(mlp(6, 16, 3), pop_size = P)
    inner_opt = pop.optimizer(torch.optim.Adam, lr = 1e-2, shared = shared)
    meta_net = mlp(6, 16, 3)
    meta_opt = torch.optim.Adam(meta_net.parameters(), lr = 1e-2)

    pp = pop.population_parameters
    ema_pp = pp.clone()
    obs1, obs2 = torch.randn(P, 8, 6), torch.randn(P, 8, 6)
    actions = torch.randint(0, 3, (P, 8))

    cur_pp = pp
    for obs in (obs1, obs2):
        logits = pop(obs, params = cur_pp, per_individual = True)
        with torch.no_grad():
            assert pop(obs, params = ema_pp, per_individual = True).shape == (P, 8, 3)
        loss = F.kl_div(logits.log_softmax(dim = -1), meta_net(obs).softmax(dim = -1), reduction = 'batchmean')
        cur_pp = inner_opt.functional_step(loss, cur_pp, create_graph = True)

    meta_opt.zero_grad()
    F.cross_entropy(pop(obs2, params = cur_pp, per_individual = True).reshape(-1, 3), actions.reshape(-1)).backward()
    meta_grad = sum(p.grad.abs().sum().item() for p in meta_net.parameters() if p.grad is not None)
    assert meta_grad > 0.
    assert torch.isfinite(torch.tensor(meta_grad))
    meta_opt.step()

    # TBPTT detach + copy_from_ + masked reset_ + lerp_
    next_pp = cur_pp.detach()
    assert next_pp.flat_params.grad_fn is None
    assert next_pp.flat_params.requires_grad
    assert next_pp.opt_slot_flat('exp_avg').grad_fn is None

    pp.copy_from_(next_pp)
    keep_before = pp.flat_params[[1, 3, 5]].detach().clone()
    mask = torch.tensor([True, False, True, False, True, False])
    pop.reset_(mask)
    ema_pp.copy_from_(pp, members = mask, include_opt = False)
    assert torch.equal(ema_pp.flat_params[mask], pp.flat_params[mask])
    ema_pp.lerp_(pp, 0.05)

    assert torch.equal(pp.flat_params[[1, 3, 5]], keep_before)
    assert not torch.allclose(pp.flat_params[0], next_pp.flat_params[0])
    if not shared:
        assert torch.equal(pp.opt_slot_flat('exp_avg')[mask], torch.zeros_like(pp.opt_slot_flat('exp_avg')[mask]))
        step_aux = pp.opt_aux('step', '0.weight')
        assert torch.equal(step_aux[mask], torch.zeros_like(step_aux[mask]))
        assert (step_aux[~mask] == 2).all()

def test_functional_step_matches_inplace():
    torch.manual_seed(7)
    nets = [mlp() for _ in range(3)]
    pop_in = Population(nets)
    pop_fn = Population([deepcopy(m) for m in nets])
    opt_in = pop_in.optimizer(torch.optim.Adam, lr = 1e-2, shared = False)
    opt_fn = pop_fn.optimizer(torch.optim.Adam, lr = 1e-2, shared = False)

    pp_fn = pop_fn.population_parameters
    for _ in range(5):
        xp, yp = torch.randn(3, 6, 8), torch.randn(3, 6, 4)
        step_pop(pop_in, opt_in, xp, yp)
        loss = (pop_fn(xp, params = pp_fn, per_individual = True) - yp).pow(2).mean(dim = (1, 2)).sum()
        pp_fn = opt_fn.functional_step(loss, pp_fn, create_graph = False)

    assert torch.allclose(pop_in.population_parameters.flat_params, pp_fn.flat_params, atol = 1e-6)
    for slot in ('exp_avg', 'exp_avg_sq'):
        assert torch.allclose(pop_in.population_parameters.opt_slot_flat(slot), pp_fn.opt_slot_flat(slot), atol = 1e-6)

def test_lifecycle_lockstep_resize_and_mean():
    pop = Population([mlp(seed = i) for i in range(4)], shared = ('0.bias',))
    opt = pop.optimizer(torch.optim.RMSprop, lr = [1e-2, 2e-2, 3e-2, 4e-2])
    pp = pop.population_parameters
    x, y = torch.randn(10, 8), torch.randn(10, 4)

    for _ in range(3):
        step_pop(pop, opt, x, y, per_individual = False)

    # mean soup
    avg = pop.mean()
    assert isinstance(avg, nn.Sequential)
    assert torch.allclose(avg[0].weight, pop.params_dict['0.weight'].mean(0), atol = 1e-6)
    assert torch.allclose(avg[0].bias, pop.params_dict['0.bias'], atol = 1e-6)

    # replace
    pop.replace(3, 0, reset_opt = True)
    assert torch.allclose(pop.params_dict['0.weight'][3], pop.params_dict['0.weight'][0])
    assert pp.opt_slot_flat('square_avg')[3].norm() == 0
    assert pp.opt_slot_flat('square_avg')[0].norm() > 0

    # delete
    w2 = pop.params_dict['0.weight'][2].detach().clone()
    pop.delete(1)
    assert pop.pop_size == 3
    assert pp.flat_params.shape[0] == 3
    assert pp.opt_slot_flat('square_avg').shape[0] == 3
    assert pp.opt_aux('step', '0.weight').shape[0] == 3
    assert torch.allclose(pop.params_dict['0.weight'][1], w2)
    assert torch.allclose(opt.param_groups[0]['lr'], torch.tensor([1e-2, 3e-2, 4e-2]))

    # keep
    pop.keep((0, 2))
    assert pop.pop_size == 2
    assert pp.flat_params.shape[0] == 2
    assert pp.opt_slot_flat('square_avg').shape[0] == 2

    # append & continue training
    fresh = mlp(seed = 99)
    assert pop.append(fresh) == 2
    assert pop.append(0) == 3
    assert pop.pop_size == 4
    assert torch.allclose(pop.params_dict['0.weight'][2], fresh[0].weight)
    assert torch.allclose(pop.params_dict['0.weight'][3], pop.params_dict['0.weight'][0])
    assert pp.opt_slot_flat('square_avg')[2].norm() == 0

    step_pop(pop, opt, x, y, per_individual = False)
    assert pp.opt_slot_flat('square_avg')[2].norm() > 0

def test_generic_opt_store():
    pop = Population([mlp(seed = i) for i in range(3)], shared = ('0.bias',))
    pp = pop.population_parameters

    store = pp.register_opt_store('custom')
    store.alloc_slot('a', [s.name for s in pp.trainable_specs if len(s.shape) >= 2])
    store.alloc_slot('b', [s.name for s in pp.trainable_specs if len(s.shape) < 2])
    store.alloc_global('t', torch.zeros(()))

    assert store.slot_views('a')['0.weight'].shape == (3, 16, 8)
    assert store.slot_views('b')['0.bias'].shape == (16,)
    assert store.slot_views('b')['2.bias'].shape == (3, 4)

    with torch.no_grad():
        store.slot_views('a')['0.weight'][1].fill_(7.0)
        store.slot_views('b')['2.bias'][2].fill_(3.0)
        store.global_tensor('t').fill_(5.0)

    pop.delete(0)
    assert store.slot_views('a')['0.weight'].shape == (2, 16, 8)
    assert torch.allclose(store.slot_views('a')['0.weight'][0], torch.full((16, 8), 7.0))
    assert store.slot_views('b')['0.bias'].shape == (16,)
    assert torch.allclose(store.slot_views('b')['2.bias'][1], torch.full((4,), 3.0))

    pop.append(0)
    assert store.slot_views('a')['0.weight'].shape == (3, 16, 8)
    assert store.slot_views('a')['0.weight'][2].norm() == 0

    store.reset_members_([0])
    assert store.slot_views('a')['0.weight'][0].norm() == 0
    assert torch.allclose(store.slot_views('b')['2.bias'][1], torch.full((4,), 3.0))

def test_detach_is_backed_and_clone_without_opt():
    torch.manual_seed(11)
    pop = Population(mlp(), pop_size = 4)
    opt = pop.optimizer(torch.optim.Adam, lr = 1e-2)
    pp = pop.population_parameters

    # clone without optimizer state for lightweight EMA shadow
    ema_pp = pp.clone(include_opt = False)
    assert ema_pp.opt_slot_flat('exp_avg') is None

    x = torch.randn(4, 6, 8)
    cur_pp = opt.functional_step(pop(x, params = pp, per_individual = True).pow(2).mean(), pp, create_graph = True)
    assert cur_pp.is_functional

    with pytest.raises(ValueError):
        opt.functional_step(pop(x, params = ema_pp, per_individual = True).pow(2).mean(), ema_pp)

    # detach returns a backed PopulationParameters supporting dtype conversion, state_dict, keep_, append_, mean()
    detached = cur_pp.detach()
    assert not detached.is_functional
    assert detached.flat_params.grad_fn is None
    assert detached.flat_params.requires_grad
    assert '_pop_param' in detached.state_dict()

    detached.double()
    assert detached.flat_params.dtype == torch.float64
    assert detached.opt_slot_flat('exp_avg').dtype == torch.float64

    detached.keep_([0, 2])
    assert detached.pop_size == 2
    assert detached.flat_params.shape[0] == 2
    assert detached.opt_slot_flat('exp_avg').shape[0] == 2

    avg_mod = detached.mean()
    assert avg_mod[0].weight.requires_grad
    assert avg_mod[0].weight.dtype == torch.float64

    # ema copy_from_ + lerp_ works cleanly without opt state
    mask = torch.tensor([True, False, True, False])
    ema_pp.copy_from_(pp, members = mask)
    ema_pp.lerp_(pp, 0.1)
    assert torch.allclose(ema_pp.flat_params[mask], pp.flat_params[mask])

    # detached + resized pp with per-individual HPs reads its own resized HPs
    pop_hp = Population(mlp(), pop_size = 4)
    opt_hp = pop_hp.optimizer(torch.optim.RMSprop, lr = [1e-4, 1e-3, 1e-2, 1e-1])
    sub_pp = pop_hp.population_parameters.detach()
    sub_pp.keep_([1, 3])
    assert torch.allclose(sub_pp.as_dict()['optimizer_state']['param_groups'][0]['lr'], torch.tensor([1e-3, 1e-1]))
    assert sub_pp[1]['optimizer_state']['param_groups'][0]['lr'] == pytest.approx(1e-1)
    stepped = opt_hp.functional_step(pop_hp(torch.randn(2, 6, 8), params = sub_pp, per_individual = True).pow(2).mean(), sub_pp, create_graph = False)
    assert stepped.pop_size == 2


def test_multi_module_params_surrogate_reconcile_and_set_grads():
    torch.manual_seed(24)

    class BranchedActor(nn.Module):
        def __init__(self):
            super().__init__()
            self.active_enc = nn.Linear(8, 8)
            self.unused_enc = nn.Linear(8, 8)
            self.head = nn.Linear(8, 4)

        def forward(self, x, pad):
            return self.head(torch.tanh(self.active_enc(x + pad)))

    class Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.actor = BranchedActor()
            self.extra_pad = nn.Linear(8, 8)
            # includes nested submodule (actor.head -> deduplicated), standalone module, and None
            self.actor_params = PopulationParameters((self.actor, self.actor.head, self.extra_pad, None))

    wrap = Wrapper()
    # zero state_dict pollution!
    assert not any(k.startswith('actor_params.') for k in wrap.state_dict())
    # deduplicated: actor.head not counted twice
    assert len(wrap.actor_params._schema.trainable) == len(list(wrap.actor.parameters())) + len(list(wrap.extra_pad.parameters()))

    def damper(g_r, g_t, eta = 0.5):
        n_r = g_r.norm(dim = -1, keepdim = True).clamp(min = 1e-8)
        n_t = g_t.norm(dim = -1, keepdim = True).clamp(min = 1e-8)
        g_t_proj = g_t - ((g_r * g_t).sum(dim = -1, keepdim = True).clamp(max = 0.) / n_r.square()) * g_r
        weight = (n_t / n_r).clamp(max = 1.).pow(1. - eta)
        return g_r + (n_r / n_t) * weight * g_t_proj

    x, x_next = torch.randn(6, 8), torch.randn(6, 8)
    pad = wrap.extra_pad(torch.ones(6, 8))
    y, y_next = wrap.actor(x, pad), wrap.actor(x_next, pad)
    loss_r = y.pow(2).mean()
    loss_t = (y_next - y).pow(2).mean()

    # reconcile returns exact value, works with composite loss & standard backward(), keeps unused branch grad = None!
    reconciled = wrap.actor_params.reconcile(loss_r, loss_t, fn = damper)
    assert torch.equal(reconciled, loss_r.detach())

    extra_term = 0.25 * wrap.extra_pad.bias.pow(2).sum()
    (reconciled + extra_term).backward()

    assert wrap.actor.active_enc.weight.grad is not None and wrap.actor.active_enc.weight.grad.abs().sum() > 0
    assert wrap.actor.head.weight.grad is not None and wrap.actor.head.weight.grad.abs().sum() > 0
    assert wrap.extra_pad.weight.grad is not None and wrap.extra_pad.weight.grad.abs().sum() > 0
    assert wrap.actor.unused_enc.weight.grad is None  # unused branch untouched!

    # eval / no_grad short-circuits cleanly
    with torch.no_grad():
        pad_ng = wrap.extra_pad(torch.ones(6, 8))
        lr_ng = wrap.actor(x, pad_ng).pow(2).mean()
        lt_ng = (wrap.actor(x_next, pad_ng) - wrap.actor(x, pad_ng)).pow(2).mean()
        out_ng = wrap.actor_params.reconcile(lr_ng, lt_ng, fn = damper)
        assert torch.equal(out_ng, lr_ng) and not out_ng.requires_grad

    # set_grads_ (overwrite & accumulate) + surrogate (populated + per-member weights + create_graph=True)
    wrap.actor_params.zero_grad_()
    g_ones = torch.ones_like(wrap.actor_params.flat_params)
    wrap.actor_params.set_grads_(g_ones)
    assert torch.allclose(wrap.actor.head.weight.grad, torch.ones_like(wrap.actor.head.weight))
    wrap.actor_params.set_grads_(2. * g_ones, accumulate = True)
    assert torch.allclose(wrap.actor.head.weight.grad, torch.full_like(wrap.actor.head.weight, 3.))

    # populated reconcile + per-member surrogate + differentiable meta-parameter eta
    pop = Population(mlp(), pop_size = 3)
    eta = nn.Parameter(torch.tensor(0.5))
    y, y_next = pop(x), pop(x_next)
    per_member_r = y.pow(2).mean(dim = (1, 2))  # (3,)
    surr = pop.reconcile(per_member_r.sum(), (y_next - y).pow(2).mean(), fn = lambda gr, gt: damper(gr, gt, eta), value = per_member_r, create_graph = True)
    assert surr.shape == (3,)
    (g_pop,) = torch.autograd.grad((surr * torch.tensor([1., 2., 3.])).sum(), (pop.population_parameters._pop_param,), create_graph = True)
    g_pop.pow(2).sum().backward()
    assert eta.grad is not None and eta.grad.isfinite()


def test_lerp_self_aliasing_and_subset_weights():
    torch.manual_seed(0)
    pop = Population(nn.Sequential(nn.Linear(4, 8), nn.BatchNorm1d(8), nn.Linear(8, 2)), pop_size = 4)
    pop(torch.randn(6, 4))  # populate BN running stats
    orig = pop.flat_params.detach().clone()

    # self-aliasing permutation lerp must read original rows, not mutated ones
    perm = [1, 2, 3, 0]
    pop.lerp_(pop.view(perm), 0.5)
    assert torch.allclose(pop.flat_params, 0.5 * (orig + orig[perm]))

    # subset view with full-(P,) per-member weights
    before = pop.flat_params.detach().clone()
    w = torch.tensor([0.1, 0.2, 0.3, 0.4])
    pop.view([0, 2]).lerp_(pop.view([2, 0]), w)
    assert torch.allclose(pop.flat_params[0], before[0].lerp(before[2], 0.1))
    assert torch.allclose(pop.flat_params[2], before[2].lerp(before[0], 0.3))
    assert torch.equal(pop.flat_params[1], before[1])


def test_noop_reset_and_copy_from_invalidate_opt_cache():
    torch.manual_seed(0)
    x = torch.randn(4, 4)
    pop = Population(nn.Linear(4, 2))
    opt = pop.optimizer(torch.optim.AdamW, lr = 1e-2)
    other = pop.clone(include_opt = True)

    for _ in range(3):
        opt.zero_grad()
        pop(x).pow(2).mean().backward()
        opt.step()

    assert opt._store.global_tensor('_init').all()
    assert opt.extract(0)[1].state[next(iter(opt.extract(0)[1].state))]['step'].item() == 3.0

    # reset_ must clear cached noop optimizer state so step restarts cleanly from 1
    pop.reset_()
    assert not opt._store.global_tensor('_init').any()
    opt.zero_grad()
    pop(x).pow(2).mean().backward()
    opt.step()
    assert opt._store.global_tensor('_init').all()
    assert opt.extract(0)[1].state[next(iter(opt.extract(0)[1].state))]['step'].item() == 1.0

    # copy_from_(other) must also drop cached noop optimizer state
    pop.copy_from_(other, include_opt = True)
    assert not opt._store.global_tensor('_init').any()


def test_multi_module_noop_guards():
    pp = PopulationParameters((nn.Linear(4, 2), nn.Linear(2, 1)))
    with pytest.raises(ValueError, match = 'single-module'):
        pp.reduce()
    with pytest.raises(ValueError, match = 'single-module'):
        pp.reset_()
