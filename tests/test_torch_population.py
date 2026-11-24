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
