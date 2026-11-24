from copy import deepcopy

import pytest
import torch
from torch import nn
from torch.nn import Module

from torch_population import Population, PopulationView

# fixtures

def mlp(dim_in = 8, dim_hidden = 16, dim_out = 4, seed = None):
    if seed is not None:
        torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(dim_in, dim_hidden),
        nn.SiLU(),
        nn.Linear(dim_hidden, dim_out),
    )

class TwoBranchNet(Module):
    def __init__(self, dim_in = 8, dim_hidden = 16, dim_out = 4):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(dim_in, dim_hidden), nn.SiLU())
        self.head = nn.Linear(dim_hidden, dim_out)
        self.register_buffer('scale', torch.tensor(1.5))
        self.register_buffer('temp', torch.tensor(0.5), persistent = False)

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x, gain = 1.0, return_hidden = False):
        h = self.encode(x)
        out = self.head(h) * self.scale * self.temp * gain
        return (out, h) if return_hidden else out

class TiedWeightNet(Module):
    def __init__(self, dim = 8):
        super().__init__()
        self.a = nn.Linear(dim, dim, bias = False)
        self.b = nn.Linear(dim, dim, bias = False)
        self.b.weight = self.a.weight

    def forward(self, x):
        return self.b(self.a(x))

# tests

def test_noop_wrapper():
    net = mlp()
    pop = Population(net)

    assert not pop.is_population
    assert pop.pop_size is None
    assert len(pop) == 1
    assert pop.unwrap() is pop.unwrap(0) is pop.unwrap(-1) is pop[0] is pop[-1] is net
    assert list(pop) == [net]
    assert all(a is b for a, b in zip(pop.parameters(), net.parameters()))

    x = torch.randn(5, 8)
    assert torch.equal(pop(x), net(x))

    opt = torch.optim.SGD(pop.parameters(), lr = 0.1)
    before = net[0].weight.detach().clone()
    pop(x).pow(2).mean().backward()
    opt.step()
    assert not torch.equal(net[0].weight, before)

    rich = TwoBranchNet()
    out, hidden = Population(rich)(x, gain = 2.0, return_hidden = True)
    ref_out, ref_hidden = rich(x, gain = 2.0, return_hidden = True)
    assert torch.equal(out, ref_out)
    assert torch.equal(hidden, ref_hidden)

    with pytest.raises(IndexError):
        _ = pop[1]

def test_populate_and_pop_size_one():
    net = mlp()
    orig_w0 = net[0].weight.detach().clone()
    x = torch.randn(6, 8)

    pop = Population(net, pop_size = 4)
    assert pop.is_population
    assert pop.pop_size == len(pop) == 4
    assert torch.equal(net[0].weight, orig_w0)

    out = pop(x)
    assert out.shape == (4, 6, 4)
    assert torch.allclose(out[0], net(x), atol = 1e-6)
    assert not torch.allclose(out[0], out[1])
    assert not torch.allclose(out[1], out[2])
    assert sum(p.numel() for p in pop.parameters()) == 4 * sum(p.numel() for p in net.parameters())

    # noop -> populate()
    noop = Population(net)
    assert noop.populate(5) is noop
    assert noop.is_population
    assert len(noop) == 5
    assert noop(x).shape == (5, 6, 4)
    assert torch.allclose(noop(x)[0], net(x), atol = 1e-6)

    with pytest.raises(ValueError):
        noop.populate(3)

    # pop_size = 1 keeps leading pop dim and unwraps cleanly
    lin = nn.Linear(8, 4)
    pop1 = Population(lin, pop_size = 1)
    assert pop1.is_population
    assert pop1(x).shape == (1, 6, 4)
    assert torch.allclose(pop1(x)[0], lin(x), atol = 1e-6)

    rec = pop1.unwrap()
    assert isinstance(rec, nn.Linear)
    assert torch.allclose(rec(x), lin(x), atol = 1e-6)

def test_init_modes_and_sources():
    net = nn.Linear(8, 4)
    x = torch.randn(5, 8)

    cloned = Population(net, 3, init = 'clone')(x)
    assert torch.allclose(cloned[0], cloned[1])
    assert torch.allclose(cloned[0], cloned[2])

    noisy = Population(net, 3, init = 'clone', noise_std = 0.1, keep_first = True)(x)
    assert torch.allclose(noisy[0], net(x), atol = 1e-6)
    assert not torch.allclose(noisy[0], noisy[1])

    noisy_all = Population(net, 3, init = 'clone', noise_std = 0.1, keep_first = False)(x)
    assert not torch.allclose(noisy_all[0], net(x))

    count = [0]
    def custom_init(m):
        count[0] += 1
        with torch.no_grad():
            m.weight.fill_(float(count[0]))
            m.bias.zero_()

    custom = Population(net, 3, init = custom_init)
    for i in range(3):
        assert torch.allclose(custom.params_dict['weight'][i], torch.full((4, 8), float(i + 1)))

    # from module list & callable factory
    nets = [nn.Linear(8, 4) for _ in range(3)]
    out = Population(nets)(x)
    assert out.shape == (3, 5, 4)
    for i, m in enumerate(nets):
        assert torch.allclose(out[i], m(x), atol = 1e-6)

    assert Population(lambda: nn.Linear(8, 4), pop_size = 4)(x).shape == (4, 5, 4)
    noop_f = Population(lambda: nn.Linear(8, 4))
    assert not noop_f.is_population
    assert noop_f(x).shape == (5, 4)

def test_per_individual_and_training():
    pop = Population(TwoBranchNet(), pop_size = 4)
    x_pop = torch.randn(4, 6, 8)
    out, hidden = pop(x_pop, gain = 2.0, return_hidden = True, per_individual = True)
    assert out.shape == (4, 6, 4)
    assert hidden.shape == (4, 6, 16)

    for i in range(4):
        ref_out, ref_h = pop[i](x_pop[i], gain = 2.0, return_hidden = True)
        assert torch.allclose(out[i], ref_out, atol = 1e-5)
        assert torch.allclose(hidden[i], ref_h, atol = 1e-5)

    with pytest.raises(ValueError):
        pop(torch.randn(3, 6, 8), per_individual = True)

    pop_t = Population(mlp(), pop_size = 4)
    opt = torch.optim.Adam(pop_t.parameters(), lr = 1e-2)
    before = [p.detach().clone() for p in pop_t.parameters()]
    (pop_t(torch.randn(10, 8)) - torch.randn(1, 10, 4)).pow(2).mean().backward()
    opt.step()
    assert all(not torch.equal(a, b) for a, b in zip(before, pop_t.parameters()))

def test_indexing_slicing_and_writeback():
    net = TwoBranchNet()
    pop = Population(net, pop_size = 4)
    x = torch.randn(5, 8)
    pop_out = pop(x)

    for idx in (0, 2, -1, torch.tensor(1)):
        ind = pop[idx]
        assert isinstance(ind, TwoBranchNet)
        assert torch.allclose(ind(x), pop_out[int(idx)], atol = 1e-5)
        assert ind.encode(x).shape == (5, 16)
        assert torch.equal(ind.temp, torch.tensor(0.5))
        assert list(ind.state_dict()) == list(net.state_dict())

    # solo training does not mutate pop until writeback
    ind = pop[2]
    before = pop.params_dict['head.weight'].detach().clone()
    opt = torch.optim.SGD(ind.parameters(), lr = 0.1)
    ind(x).pow(2).mean().backward()
    opt.step()
    assert torch.equal(pop.params_dict['head.weight'], before)

    pop[2] = ind
    assert torch.allclose(pop(x)[2], ind(x), atol = 1e-5)
    assert torch.equal(pop.params_dict['head.weight'][0], before[0])

    assert torch.allclose(Population(ind, pop_size = 3)(x)[0], ind(x), atol = 1e-5)
    assert len(list(pop)) == 4
    assert all(isinstance(m, TwoBranchNet) for m in pop)

    # slice, list (with duplicates), bool mask & slice writeback
    pop5 = Population(nn.Linear(8, 4), pop_size = 5)
    full = pop5(x)
    mask = torch.tensor([True, False, True, False, True])

    assert torch.allclose(pop5[1:4](x), full[1:4], atol = 1e-6)
    assert torch.allclose(pop5[[3, 0, 3]](x), full[[3, 0, 3]], atol = 1e-6)
    assert torch.allclose(pop5[mask](x), full[mask], atol = 1e-6)

    pop5[0:2] = pop5[3:5]
    assert torch.allclose(pop5(x)[0:2], full[3:5], atol = 1e-6)

def test_live_views():
    pop = Population(mlp(), pop_size = 4)
    x = torch.randn(6, 8)

    v1 = pop.view(1)
    assert isinstance(v1, PopulationView)
    assert v1.is_single
    assert len(v1) == 1

    out1 = v1(x)
    assert out1.shape == (6, 4)
    assert torch.allclose(out1, pop(x)[1], atol = 1e-5)

    out1.pow(2).mean().backward()
    g = pop.grads_dict['0.weight']
    assert g[1].abs().sum() > 0
    assert all(torch.equal(g[i], torch.zeros_like(g[i])) for i in (0, 2, 3))

    pop.zero_grad()
    v_sub = pop.view([0, 3])
    assert not v_sub.is_single
    assert len(v_sub) == 2

    v_sub(x).pow(2).mean().backward()
    g = pop.grads_dict['0.weight']
    assert g[0].abs().sum() > 0
    assert g[3].abs().sum() > 0
    assert torch.equal(g[1], torch.zeros_like(g[1]))
    assert torch.equal(g[2], torch.zeros_like(g[2]))

    x_sub = torch.randn(2, 5, 8)
    out_sub = pop.view([1, 3])(x_sub, per_individual = True)
    assert torch.allclose(out_sub[0], pop[1](x_sub[0]), atol = 1e-5)
    assert torch.allclose(out_sub[1], pop[3](x_sub[1]), atol = 1e-5)

def test_shared_parameters_and_subset_forward():
    models = [mlp(seed = i) for i in range(4)]
    pop = Population(models, shared = ('0.bias',))
    opt = pop.optimizer(torch.optim.RMSprop, lr = 1e-2)
    pp = pop.population_parameters

    assert pp.is_shared_param('0.bias')
    assert not pp.is_shared_param('0.weight')
    assert pop.params_dict['0.bias'].shape == (16,)
    assert pop.params_dict['0.weight'].shape == (4, 16, 8)
    assert pp.flat_shared_params.shape == (16,)
    assert torch.allclose(pop.params_dict['0.bias'], torch.stack([m[0].bias.detach() for m in models]).mean(0))

    x, y = torch.randn(12, 8), torch.randn(12, 4)
    w0 = pop.params_dict['0.weight'].detach().clone()
    b0 = pop.params_dict['0.bias'].detach().clone()

    opt.zero_grad()
    (pop(x) - y.unsqueeze(0)).pow(2).mean().backward()
    opt.step()
    assert not torch.allclose(pop.params_dict['0.weight'], w0)
    assert not torch.allclose(pop.params_dict['0.bias'], b0)

    # subset forward & subset step
    sub = pop(x, members = (0, 2))
    assert sub.shape == (2, 12, 4)
    assert torch.allclose(sub[0], pop[0](x), atol = 1e-5)
    assert torch.allclose(sub[1], pop[2](x), atol = 1e-5)
    assert torch.allclose(pop(x, members = torch.tensor([False, True, False, True]))[0], pop[1](x), atol = 1e-5)

    w1 = pop.params_dict['0.weight'].detach().clone()
    opt.zero_grad()
    (pop(x, members = (0, 2)) - y.unsqueeze(0)).pow(2).mean().backward()
    opt.step(members = (0, 2))
    assert not torch.allclose(pop.params_dict['0.weight'][0], w1[0])
    assert not torch.allclose(pop.params_dict['0.weight'][2], w1[2])
    assert torch.allclose(pop.params_dict['0.weight'][1], w1[1])
    assert torch.allclose(pop.params_dict['0.weight'][3], w1[3])

    ind, ind_opt = pop.extract(1)
    assert isinstance(ind_opt, torch.optim.RMSprop)
    assert torch.allclose(ind[0].bias, pop.params_dict['0.bias'])
    assert torch.allclose(ind[0].weight, pop.params_dict['0.weight'][1])
    assert '0.bias' in pop.individual_dict(1)['named_optimizer_state']

    pop_s = Population(mlp(seed = 0), 3, init = 'clone', shared = ('0.bias',))
    assert pop_s.params_dict['0.bias'].shape == (16,)
    assert torch.allclose(pop_s.params_dict['0.weight'][0], pop_s.params_dict['0.weight'][2])

def test_batchnorm_dropout_nested_and_noop_setitem():
    net = nn.Sequential(nn.Linear(8, 16), nn.BatchNorm1d(16), nn.Dropout(0.5), nn.ReLU(), nn.Linear(16, 4))
    pop = Population(net, pop_size = 3)
    x_pop = torch.randn(3, 8, 8) + torch.arange(3).view(3, 1, 1) * 5.0

    pop.train()
    pop(x_pop, per_individual = True).sum().backward()
    rm = pop.buffers_dict['1.running_mean']
    assert not torch.allclose(rm[0], rm[1])

    pop.eval()
    x = torch.randn(4, 8)
    out1, out2 = pop(x), pop(x)
    assert torch.equal(out1, out2)

    ind1 = pop[1]
    assert not ind1.training
    assert torch.allclose(ind1(x), out1[1], atol = 1e-5)

    # nested in parent Module & noop __setitem__
    parent = nn.Sequential(Population(nn.Sequential(nn.Linear(8, 16), nn.Dropout(0.5), nn.Linear(16, 4)), 3))
    parent.eval()
    assert torch.equal(parent(x), parent(x))

    base, other = nn.Linear(8, 4), nn.Linear(8, 4)
    noop = Population(base)
    noop[0] = other
    assert noop.unwrap() is base
    assert torch.allclose(noop(x), other(x))
    assert torch.allclose(noop.view(0)(x), other(x))

def test_tied_frozen_parameterless_state_dict_and_dtype():
    x = torch.randn(4, 8)

    # tied weights
    tied = TiedWeightNet(8)
    pop_t = Population(tied, pop_size = 3)
    assert len(pop_t.params_dict) == 1
    assert torch.allclose(pop_t(x)[0], tied(x), atol = 1e-5)
    ind_t = pop_t[2]
    assert ind_t.a.weight is ind_t.b.weight
    assert torch.allclose(ind_t(x), pop_t(x)[2], atol = 1e-5)

    # frozen parameter
    lin = nn.Linear(8, 4)
    lin.bias.requires_grad_(False)
    pop_f = Population(lin, pop_size = 3)
    assert pop_f.params_dict['weight'].requires_grad
    assert not pop_f.params_dict['bias'].requires_grad
    assert pop_f[1].weight.requires_grad
    assert not pop_f[1].bias.requires_grad

    # parameterless modules
    out_id = Population(nn.Identity(), pop_size = 3)(x)
    assert out_id.shape == (3, 4, 8)
    assert torch.equal(out_id[0], x)
    out_drop = Population(nn.Dropout(0.5), pop_size = 4)(torch.randn(4, 16))
    assert out_drop.shape == (4, 4, 16)
    assert not torch.equal(out_drop[0], out_drop[1])

    # state_dict & dtype conversion
    pop1, pop2 = Population(TwoBranchNet(), 3), Population(TwoBranchNet(), 3)
    assert not torch.allclose(pop1(x), pop2(x))
    pop2.load_state_dict(pop1.state_dict())
    assert torch.allclose(pop1(x), pop2(x), atol = 1e-6)

    pop64 = pop1.double()
    out64 = pop64(x.double())
    assert out64.dtype == torch.float64
    assert pop64[0].head.weight.dtype == torch.float64
    assert torch.allclose(pop64[0](x.double()), out64[0], atol = 1e-10)

def test_per_individual_default_requires_grad_and_selectors():
    torch.manual_seed(0)
    lin = nn.Linear(8, 4)
    lin.bias.requires_grad_(False)
    pop = Population(lin, pop_size = 4, per_individual = True)

    x_pop = torch.randn(4, 5, 8)
    out = pop(x_pop)
    assert out.shape == (4, 5, 4)
    assert torch.allclose(out[2], pop[2](x_pop[2]), atol = 1e-6)

    # override per_individual=False on a single call and preserve default across slicing
    x_shared = torch.randn(5, 8)
    assert pop(x_shared, per_individual = False).shape == (4, 5, 4)
    sub = pop[[True, False, True, False]]
    assert sub.pop_size == 2
    assert sub.per_individual
    assert sub(torch.randn(2, 5, 8)).shape == (2, 5, 4)

    with pytest.raises(TypeError):
        _ = pop[True]
    with pytest.raises(ValueError):
        _ = pop[torch.tensor(True)]
    with pytest.raises(TypeError):
        _ = pop.population_parameters[torch.tensor(True)]

    # self-permuting slice assignment & view source replace
    w_before = pop.params_dict['weight'].detach().clone()
    pop[[1, 0, 3, 2]] = pop
    assert torch.equal(pop.params_dict['weight'][0], w_before[1])
    assert torch.equal(pop.params_dict['weight'][1], w_before[0])
    pop.replace(2, pop.view(0))
    assert torch.equal(pop.params_dict['weight'][2], pop.params_dict['weight'][0])
    with pytest.raises(ValueError):
        pop.replace(2, pop.view([0, 1]))

    pop[0:2] = pop.view([2, 3])
    assert torch.equal(pop.params_dict['weight'][0], pop.params_dict['weight'][2])
    assert torch.equal(pop.params_dict['weight'][1], pop.params_dict['weight'][3])

    # requires_grad preserved under no_grad() and in mean()
    with torch.no_grad():
        ind = pop[1]
    assert ind.weight.requires_grad
    assert not ind.bias.requires_grad

    avg = pop.mean()
    assert avg.weight.requires_grad
    assert not avg.bias.requires_grad

    # pattern matching is exact / module-prefix / glob (not naive substring)
    deep = nn.Sequential(*[nn.Linear(4, 4) for _ in range(11)])
    pop_deep = Population(deep, pop_size = 2, shared = ('0.bias',))
    assert pop_deep.population_parameters.is_shared_param('0.bias')
    assert not pop_deep.population_parameters.is_shared_param('10.bias')

def test_call_arbitrary_methods_shared_buffers_and_reduce():
    torch.manual_seed(21)
    net = TwoBranchNet()
    x = torch.randn(5, 8)

    # noop call by method name & callable
    noop = Population(net)
    assert torch.equal(noop.call('encode', x), net.encode(x))
    assert torch.equal(noop.call(lambda m, t: m.encode(t) * 2., x), net.encode(x) * 2.)

    # population call with shared submodule + shared buffer
    pop = Population(net, pop_size = 4, shared = ('encoder', 'scale'))
    pp = pop.population_parameters
    assert pp.is_shared_param('encoder.0.weight')
    assert pp.is_shared_buffer('scale')
    assert not pp.is_shared_buffer('temp')
    assert pop.buffers_dict['scale'].shape == ()
    assert pop.buffers_dict['temp'].shape == (4,)

    h = pop.call('encode', x)
    assert h.shape == (4, 5, 16)
    assert torch.allclose(h[0], h[3], atol = 1e-6)
    assert torch.allclose(h[1], pop[1].encode(x), atol = 1e-6)

    # subset, per_individual, callable & live view call
    x_sub = torch.randn(2, 5, 8)
    h_sub = pop.call('encode', x_sub, members = (1, 3), per_individual = True)
    assert h_sub.shape == (2, 5, 16)
    assert torch.allclose(h_sub[0], pop[1].encode(x_sub[0]), atol = 1e-6)
    assert torch.allclose(h_sub[1], pop[3].encode(x_sub[1]), atol = 1e-6)

    scaled = pop.call(lambda m, t: m.head(m.encode(t)) * m.scale, x)
    for i in range(4):
        assert torch.allclose(scaled[i], pop[i].head(pop[i].encode(x)) * pop[i].scale, atol = 1e-5)

    assert torch.allclose(pop.view(2).call('encode', x), pop[2].encode(x), atol = 1e-6)
    assert torch.allclose(pop.view([0, 2]).call('encode', x)[1], pop[2].encode(x), atol = 1e-6)

    # lifecycle preserves shared buffer shape
    pop.reset_(torch.tensor([True, False, False, False]))
    pop.delete(1)
    pop.append(0)
    assert pop.buffers_dict['scale'].shape == ()
    assert pop.buffers_dict['temp'].shape == (4,)

    # generalized pop.reduce (median, max, custom callable)
    med = pop.reduce('median')
    assert torch.allclose(med.head.weight, pop.params_dict['head.weight'].median(dim = 0).values)
    assert torch.allclose(med.scale, pop.buffers_dict['scale'])
    hi = pop.reduce('max')
    assert torch.allclose(hi.head.weight, pop.params_dict['head.weight'].amax(dim = 0))
    custom = pop.reduce(lambda t: 0.25 * t[0] + 0.75 * t[-1])
    assert torch.allclose(custom.head.weight, 0.25 * pop.params_dict['head.weight'][0] + 0.75 * pop.params_dict['head.weight'][-1])


def test_clone_lerp_and_ema_noop_and_populated():
    torch.manual_seed(22)
    x = torch.randn(6, 8)

    # 1. noop clone (include_opt=False/True), lerp_, and tracked grad EMA + lifecycle
    net = mlp()
    pop = Population(net)
    opt = pop.optimizer(torch.optim.AdamW, lr = 1e-2)

    ema = pop.clone(include_opt = False)
    assert not ema.is_population and ema.unwrap() is not net and ema.opt is None

    w0 = pop.flat_params.detach().clone()
    g1 = opt.autograd(pop(x).pow(2).mean(), flat = True)
    g_ema = opt.ema_('grad', g1, decay = 0.9)
    assert torch.allclose(g_ema, g1)  # first obs initializes directly (no zero bias)
    opt.step(grads = g1 + 0.5 * g_ema)

    w1 = pop.flat_params.detach().clone()
    ema.lerp_(pop, 0.25)
    assert torch.allclose(ema.flat_params, w0.lerp(w1, 0.25), atol = 1e-6)

    g2 = opt.autograd(pop(x).pow(2).mean(), flat = True)
    g_ema = pop.ema_('grad', g2, decay = 0.9)
    assert torch.allclose(g_ema, g1.lerp(g2, 0.1), atol = 1e-6)

    # noop clone with include_opt=True forks independent optimizer & tracked buffers
    fork = pop.clone(include_opt = True)
    assert fork.opt is not None and fork.opt is not opt
    assert torch.allclose(fork.tracked('grad'), g_ema)
    fork.opt.step(grads = fork.opt.autograd(fork(x).pow(2).mean(), flat = True))
    assert not torch.allclose(fork.flat_params, pop.flat_params)

    # noop dtype sync on ema_ + populate carries tracked EMAs
    net.double()
    g_double = opt.autograd(pop(x.double()).pow(2).mean(), flat = True)
    assert pop.ema_('grad', g_double, decay = 0.5).dtype == torch.float64
    net.float()
    pop.ema_('grad', torch.ones_like(g1), decay = 0.0)
    pop.populate(3, init = 'reinit', keep_first = True)
    assert pop.tracked('grad').shape == (3, g1.numel())
    assert torch.allclose(pop.tracked('grad')[0], torch.ones_like(g1))
    # newly reinit slots 1,2 have init=False so next ema_ initializes them cleanly without zero-bias
    obs = torch.stack([torch.full_like(g1, 3.), torch.full_like(g1, 5.), torch.full_like(g1, 7.)])
    pop.ema_('grad', obs, decay = 0.5)
    assert torch.allclose(pop.tracked('grad')[0], torch.full_like(g1, 2.))  # 0.5 * 1 + 0.5 * 3
    assert torch.allclose(pop.tracked('grad')[1], torch.full_like(g1, 5.))  # fresh init!
    assert torch.allclose(pop.tracked('grad')[2], torch.full_like(g1, 7.))  # fresh init!

    # 2. populated multi-timescale lerp_, subset lerp_, per-member ema_ decays & full lifecycle
    pop4 = Population(mlp, pop_size = 4)
    opt4 = pop4.optimizer(torch.optim.AdamW, lr = 1e-2)
    ema4 = pop4.clone(include_opt = False)

    g = opt4.autograd(pop4(x).pow(2).mean(), flat = True)
    opt4.step(grads = g)

    # per-member multi-timescale lerp_ + subset lerp_
    w_ema_before = ema4.flat_params.detach().clone()
    w_online = pop4.flat_params.detach().clone()
    taus = torch.tensor([0.0, 0.25, 0.5, 1.0])
    ema4.lerp_(pop4, taus)
    for i, tau in enumerate(taus.tolist()):
        assert torch.allclose(ema4.flat_params[i], w_ema_before[i].lerp(w_online[i], tau), atol = 1e-6)

    ema4.lerp_(pop4, 0.5, members = (0, 2))
    assert torch.allclose(ema4.flat_params[1], w_ema_before[1].lerp(w_online[1], 0.25), atol = 1e-6)  # untouched

    # per-member decay on ema_, subset members, reset_, replace, delete, append, state_dict, slice
    decays = torch.tensor([0.5, 0.8, 0.9, 0.99])
    v0 = torch.ones_like(g)
    v1 = torch.full_like(g, 11.)
    pop4.ema_('g', v0, decay = decays)
    pop4.ema_('g', v1, decay = decays, members = (0, 2))
    assert torch.allclose(pop4.tracked('g')[0], torch.full_like(g[0], 6.))     # 1 -> 11 @ 0.5
    assert torch.allclose(pop4.tracked('g')[1], torch.ones_like(g[1]))         # untouched
    assert torch.allclose(pop4.tracked('g')[2], torch.full_like(g[2], 2.))     # 1 -> 11 @ 0.9

    # reset_(1) clears slot 1 so its next observation initializes fresh (no zero-bias)
    pop4.reset_(1)
    assert pop4.tracked('g')[1].abs().sum() == 0
    pop4.ema_('g', torch.full_like(g, 4.), decay = 0.9)
    assert torch.allclose(pop4.tracked('g')[1], torch.full_like(g[1], 4.))

    # state_dict round-trip + slice + delete + append
    sd = {k: v.clone() for k, v in pop4.state_dict().items()}
    pop4_copy = pop4.clone(include_opt = False)
    pop4_copy.clear_tracked('g')
    assert pop4_copy.tracked('g') is None
    pop4.replace(3, 0, reset_opt = False)
    assert torch.allclose(pop4.tracked('g')[3], pop4.tracked('g')[0])
    sub = pop4[:2]
    assert sub.tracked('g').shape == (2, g.shape[-1])
    pop4.delete(1)
    assert pop4.tracked('g').shape == (3, g.shape[-1])
    pop4.append(0)
    assert pop4.tracked('g').shape == (4, g.shape[-1])
    # newly appended slot 3 has init=False -> takes fresh obs directly
    pop4.ema_('g', torch.full((4, g.shape[-1]), 9.), decay = 0.9)
    assert torch.allclose(pop4.tracked('g')[3], torch.full_like(g[0], 9.))
    assert 'population_parameters.store.k/track/g' in sd


def test_flat_params_dispatch_sam_es_soup_and_colocated_param_ema():
    torch.manual_seed(23)
    x = torch.randn(6, 8)

    # 1. SAM in 4 lines — exact parity with manual perturb-and-grad (noop AND populated)
    for pop_size in (None, 3):
        net = TwoBranchNet()
        pop = Population(net, pop_size = pop_size, shared = ('encoder',) if pop_size else ())
        opt = pop.optimizer(torch.optim.SGD, lr = 1e-2)
        rho = 0.05

        g = opt.autograd(pop(x).pow(2).mean(), flat = True)
        eps = rho * g / g.norm(dim = -1, keepdim = True).clamp(min = 1e-8)
        g_sam = opt.autograd(pop(x, params = pop.flat_params + eps.detach()).pow(2).mean(), flat = True)

        # verify against explicit perturbation on a clone
        ref = pop.clone(include_opt = False)
        with torch.no_grad():
            if pop_size is None:
                for p, e in zip((p for p in ref.unwrap().parameters() if p.requires_grad), ref.population_parameters.unpack_flat(eps).values()):
                    p.add_(e)
            else:
                ref.population_parameters._pop_param.add_(eps)
        g_ref = ref.autograd(ref(x).pow(2).mean(), flat = True)
        assert torch.allclose(g_sam, g_ref, atol = 1e-5)

        # colocated flat param EMA runnable directly via params=p_ema
        p_ema = pop.ema_('param', pop.flat_params, decay = 0.99)
        opt.step(grads = g_sam)
        out_ema = pop(x, params = p_ema)
        assert out_ema.shape == pop(x).shape

    # 2. Vectorized Evolution Strategies (K=8 perturbations on a single noop model!)
    net = mlp()
    pop = Population(net)
    opt = pop.optimizer(torch.optim.AdamW, lr = 1e-2)
    D = pop.flat_params.numel()
    sigma = 0.05
    noise = torch.randn(8, D)
    thetas = pop.flat_params.detach() + sigma * noise  # (8, D)

    outs = pop(x, params = thetas)  # vmapped across 8 perturbations!
    assert outs.shape == (8, 6, 4)

    # exact parity with manual forward on each perturbed copy
    for k in range(8):
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

    # 3. Differentiable 1D model soup evaluation on a populated Population
    pop3 = Population(mlp, pop_size = 3)
    soup_theta = pop3.flat_params.mean(dim = 0)  # (D,)
    soup_out = pop3(x, params = soup_theta)
    assert soup_out.shape == (6, 4)
    assert torch.allclose(soup_out, pop3.mean()(x), atol = 1e-5)
    soup_out.pow(2).mean().backward()
    assert pop3.population_parameters._pop_param.grad is not None
    assert pop3.population_parameters._pop_param.grad.abs().sum() > 0


def test_clip_grad_norm_noop_populated_subset_names_and_shared():
    torch.manual_seed(24)
    x = torch.randn(6, 8)

    # 1. Noop parity with torch.nn.utils.clip_grad_norm_ across norm types & names
    for norm_type in (1.0, 2.0, float('inf')):
        net_a = TwoBranchNet()
        net_b = deepcopy(net_a)
        pop_noop = Population(net_a)

        pop_noop(x).pow(2).sum().backward()
        net_b(x).pow(2).sum().backward()

        n_pop = pop_noop.clip_grad_norm_(0.25, norm_type = norm_type)
        n_ref = torch.nn.utils.clip_grad_norm_(net_b.parameters(), 0.25, norm_type = norm_type)
        assert n_pop.ndim == 0 and torch.allclose(n_pop, n_ref, atol = 1e-6)
        for pa, pb in zip(net_a.parameters(), net_b.parameters()):
            assert torch.allclose(pa.grad, pb.grad, atol = 1e-6)

    # 2. Populated per-individual independence (no 1/sqrt(P) shrinkage or cross-member interference)
    scales = torch.tensor([1e-3, 1.0, 50.0, 0.0])
    pop = Population(TwoBranchNet, pop_size = 4)
    refs = [pop[i] for i in range(4)]

    pop.zero_grad()
    (pop(x).pow(2).mean(dim = (-2, -1)) * scales).sum().backward()
    for m, s in zip(refs, scales):
        (m(x).pow(2).mean() * s).backward()

    norms = pop.clip_grad_norm_(0.5)
    ref_norms = torch.stack([torch.nn.utils.clip_grad_norm_(m.parameters(), 0.5) for m in refs])
    assert norms.shape == (4,)
    assert torch.allclose(norms, ref_norms, atol = 1e-5)
    for i, m in enumerate(refs):
        for name, p in m.named_parameters():
            assert torch.allclose(pop.grads_dict[name][i], p.grad, atol = 1e-5)

    # 3. Subset members, single/multi PopulationView, empty mask, and PopulationOptimizer forwarding
    opt = pop.optimizer(torch.optim.SGD, lr = 1e-2)
    opt.zero_grad()
    (pop(x).pow(2).mean(dim = (-2, -1)) * torch.tensor([10., 10., 10., 10.])).sum().backward()
    g_before = pop.population_parameters._pop_param.grad.clone()

    sub_norms = opt.clip_grad_norm_(0.1, members = [0, 2])
    assert sub_norms.shape == (2,)
    assert torch.equal(pop.population_parameters._pop_param.grad[1], g_before[1])
    assert torch.equal(pop.population_parameters._pop_param.grad[3], g_before[3])
    assert pop.population_parameters._pop_param.grad[0].norm() <= 0.1 + 1e-5
    assert pop.population_parameters._pop_param.grad[2].norm() <= 0.1 + 1e-5

    single_norm = pop.view(1).clip_grad_norm_(0.1)
    assert single_norm.ndim == 0
    assert pop.population_parameters._pop_param.grad[1].norm() <= 0.1 + 1e-5
    assert pop.clip_grad_norm_(0.1, members = torch.zeros(4, dtype = torch.bool)).numel() == 0

    # 4. Per-submodule scoping (`names=`) — huge head grad must not crush encoder grad
    pop.zero_grad()
    out, hidden = pop(x, return_hidden = True)
    (hidden.pow(2).mean() + 1000. * out.pow(2).mean()).backward()
    head_grad_before = pop.grads_dict['head.weight'].clone()
    enc_norms = pop.clip_grad_norm_(0.2, names = 'encoder')
    assert enc_norms.shape == (4,)
    assert torch.equal(pop.grads_dict['head.weight'], head_grad_before)
    assert pop.grads_dict['encoder.0.weight'].flatten(1).norm(dim = -1).max() <= 0.2 + 1e-5

    # 5. Shared backbone + independent heads + error_if_nonfinite
    pop_sh = Population(TwoBranchNet, pop_size = 3, shared = ('encoder',))
    pop_sh.zero_grad()
    (pop_sh(x).pow(2).mean(dim = (-2, -1)) * torch.tensor([10., 20., 30.])).sum().backward()
    pop_sh.clip_grad_norm_(0.25)
    assert pop_sh.population_parameters._pop_param.grad.norm(dim = -1).max() <= 0.25 + 1e-5
    assert pop_sh.population_parameters._shared_param.grad.norm() <= 0.25 + 1e-5

    pop_sh.population_parameters._pop_param.grad[1, 0] = float('nan')
    # clipping only member 0 ignores NaN on member 1
    assert pop_sh.clip_grad_norm_(0.25, members = [0], error_if_nonfinite = True).isfinite().all()
    with pytest.raises(RuntimeError):
        pop_sh.clip_grad_norm_(0.25, error_if_nonfinite = True)

    # no-grad safety
    pop.zero_grad()
    assert torch.equal(pop.clip_grad_norm_(1.0), torch.zeros(4))


def test_track_dtype_agnostic_and_full_lifecycle():
    torch.manual_seed(25)

    # 1. Noop -> populate carries integer / bool tracked metadata
    pop = Population(mlp())
    pop.track_('age', 5)
    pop.track_('elite', True)
    assert pop.tracked('age').dtype == torch.int64 and pop.tracked('age').item() == 5
    assert pop.tracked('elite').dtype == torch.bool and pop.tracked('elite').item() is True

    pop.populate(4, init = 'reinit', keep_first = True)
    assert pop.tracked('age').tolist() == [5, 0, 0, 0]
    assert pop.tracked('elite').tolist() == [True, False, False, False]

    # 2. Full pop, subset, PopulationView, and scalar/multi-dim dtypes
    pop.track_('age', torch.tensor([1, 2, 3, 4]))
    pop.track_('parents', torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0]]))
    pop.track_('generation', 7)  # non-populated global scalar
    assert pop.tracked('generation').ndim == 0 and pop.tracked('generation').item() == 7

    assert pop.view(2).track_('age', 99).item() == 99
    assert pop.tracked('age').tolist() == [1, 2, 99, 4]
    pop.view([0, 3]).track_('elite', True)
    assert pop.tracked('elite').tolist() == [True, False, False, True]
    pop.track_('age', torch.tensor([10, 20]), members = [1, 3])
    assert pop.tracked('age').tolist() == [1, 10, 99, 20]

    # 3. Lifecycle lockstep: keep (reorder), delete, append, reset_, replace, slice, clone, state_dict
    pop.keep([2, 0, 3, 1])
    assert pop.tracked('age').tolist() == [99, 1, 20, 10]
    assert pop.tracked('elite').tolist() == [False, True, True, False]
    assert pop.tracked('parents')[0].tolist() == [2, 3]
    assert pop.tracked('generation').item() == 7

    pop.reset_(1)
    assert pop.tracked('age')[1].item() == 0 and pop.tracked('elite')[1].item() is False

    pop.replace(1, 0, reset_opt = False)
    assert pop.tracked('age')[1].item() == 99 and pop.tracked('parents')[1].tolist() == [2, 3]

    sub = pop[:2]
    assert sub.tracked('age').tolist() == [99, 99]

    pop.delete(3)
    assert pop.tracked('age').tolist() == [99, 99, 20]
    pop.append(0)
    assert pop.tracked('age').tolist() == [99, 99, 20, 0]

    # state_dict round-trip
    sd = {k: v.clone() for k, v in pop.state_dict().items()}
    clone = pop.clone(include_opt = False)
    clone.track_('age', torch.zeros(4, dtype = torch.long))
    clone.load_state_dict(sd)
    assert clone.tracked('age').tolist() == [99, 99, 20, 0]

    # 4. Interop between track_ and ema_ on the same key
    pop.track_('score', torch.tensor([2., 4., 6., 8.]))
    pop.ema_('score', torch.tensor([6., 8., 10., 12.]), decay = 0.5)
    assert torch.allclose(pop.tracked('score'), torch.tensor([4., 6., 8., 10.]))
