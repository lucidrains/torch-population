import pytest
import torch
from torch import nn
from torch.nn import Module, Parameter

from torch_population import Population, PopulationView

param = pytest.mark.parametrize


class TwoBranchNet(Module):
    def __init__(self, dim_in = 8, dim_hidden = 16, dim_out = 4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(dim_in, dim_hidden),
            nn.SiLU(),
        )
        self.head = nn.Linear(dim_hidden, dim_out)
        self.register_buffer('scale', torch.tensor(1.5))
        self.register_buffer('temp', torch.tensor(0.5), persistent = False)

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x, gain = 1.0, return_hidden = False):
        h = self.encode(x)
        out = self.head(h) * self.scale * self.temp * gain
        if return_hidden:
            return out, h
        return out


class TiedWeightNet(Module):
    def __init__(self, dim = 8):
        super().__init__()
        self.a = nn.Linear(dim, dim, bias = False)
        self.b = nn.Linear(dim, dim, bias = False)
        self.b.weight = self.a.weight

    def forward(self, x):
        return self.b(self.a(x))


# 1. Bare minimum noop wrapper & recovering the network

def test_noop_wrapper_and_recover():
    net = nn.Sequential(nn.Linear(8, 16), nn.SiLU(), nn.Linear(16, 4))
    pop = Population(net)

    assert not pop.is_population
    assert pop.pop_size is None
    assert len(pop) == 1
    assert pop.unwrap() is net
    assert pop.unwrap(0) is net
    assert pop.unwrap(-1) is net
    assert pop[0] is net
    assert pop[-1] is net
    assert list(pop) == [net]

    # exact parameter aliasing (no copy)
    assert all(a is b for a, b in zip(pop.parameters(), net.parameters()))

    x = torch.randn(5, 8)
    out = pop(x)
    assert out.shape == (5, 4)
    assert torch.equal(out, net(x))

    # gradients flow directly to net's parameters
    opt = torch.optim.SGD(pop.parameters(), lr = 0.1)
    before = net[0].weight.detach().clone()
    pop(x).pow(2).mean().backward()
    opt.step()
    assert not torch.equal(net[0].weight, before)

    # arbitrary kwargs pass straight through
    rich = TwoBranchNet()
    pop_rich = Population(rich)
    out, hidden = pop_rich(x, gain = 2.0, return_hidden = True)
    ref_out, ref_hidden = rich(x, gain = 2.0, return_hidden = True)
    assert torch.equal(out, ref_out)
    assert torch.equal(hidden, ref_hidden)

    with pytest.raises(IndexError):
        _ = pop[1]


# 2. Turning a single network into an ensemble / population

def test_population_from_single_net_and_populate():
    net = nn.Sequential(nn.Linear(8, 16), nn.SiLU(), nn.Linear(16, 4))
    orig_w0 = net[0].weight.detach().clone()

    # constructor form
    pop = Population(net, pop_size = 4)
    assert pop.is_population
    assert pop.pop_size == 4
    assert len(pop) == 4

    # original bare network is never mutated
    assert torch.equal(net[0].weight, orig_w0)

    # member 0 matches original net by default (keep_first=True), others are independently reinitialized
    x = torch.randn(6, 8)
    out = pop(x)
    assert out.shape == (4, 6, 4)
    assert torch.allclose(out[0], net(x), atol = 1e-6)
    assert not torch.allclose(out[0], out[1])
    assert not torch.allclose(out[1], out[2])

    # only stacked population parameters are registered (no duplicate template params)
    pop_param_count = sum(p.numel() for p in pop.parameters())
    net_param_count = sum(p.numel() for p in net.parameters())
    assert pop_param_count == 4 * net_param_count

    # start as noop wrapper, then turn into a population via .populate()
    noop = Population(net)
    assert not noop.is_population
    ret = noop.populate(5)
    assert ret is noop
    assert noop.is_population
    assert len(noop) == 5
    assert noop(x).shape == (5, 6, 4)
    assert torch.allclose(noop(x)[0], net(x), atol = 1e-6)

    with pytest.raises(ValueError):
        noop.populate(3)


def test_population_size_one_keeps_pop_dim_and_matches_net():
    net = nn.Linear(8, 4)
    pop = Population(net, pop_size = 1)
    x = torch.randn(5, 8)

    assert pop.is_population
    assert len(pop) == 1
    out = pop(x)
    assert out.shape == (1, 5, 4)
    assert torch.allclose(out[0], net(x), atol = 1e-6)

    recovered = pop.unwrap()
    assert isinstance(recovered, nn.Linear)
    assert torch.allclose(recovered(x), net(x), atol = 1e-6)


def test_init_modes_clone_noise_custom_and_keep_first():
    net = nn.Linear(8, 4)
    x = torch.randn(5, 8)

    # exact clone
    cloned = Population(net, 3, init = 'clone')
    out_c = cloned(x)
    assert torch.allclose(out_c[0], out_c[1])
    assert torch.allclose(out_c[0], out_c[2])

    # clone + noise, keeping member 0 as anchor
    noisy = Population(net, 3, init = 'clone', noise_std = 0.1, keep_first = True)
    assert torch.allclose(noisy(x)[0], net(x), atol = 1e-6)
    assert not torch.allclose(noisy(x)[0], noisy(x)[1])

    # clone + noise on all members
    noisy_all = Population(net, 3, init = 'clone', noise_std = 0.1, keep_first = False)
    assert not torch.allclose(noisy_all(x)[0], net(x))

    # custom init callable
    counter = {'val': 0}

    def custom_init(m):
        counter['val'] += 1
        with torch.no_grad():
            m.weight.fill_(float(counter['val']))
            m.bias.zero_()

    custom = Population(net, 3, init = custom_init)
    assert torch.allclose(custom.params_dict['weight'][0], torch.full((4, 8), 1.0))
    assert torch.allclose(custom.params_dict['weight'][1], torch.full((4, 8), 2.0))
    assert torch.allclose(custom.params_dict['weight'][2], torch.full((4, 8), 3.0))


def test_population_from_modules_and_factory():
    nets = [nn.Linear(8, 4) for _ in range(3)]
    pop = Population(nets)
    x = torch.randn(5, 8)

    out = pop(x)
    assert out.shape == (3, 5, 4)
    for i, net in enumerate(nets):
        assert torch.allclose(out[i], net(x), atol = 1e-6)

    # factory -> population
    pop_factory = Population(lambda: nn.Linear(8, 4), pop_size = 4)
    assert pop_factory.is_population
    assert pop_factory(x).shape == (4, 5, 4)

    # factory with pop_size=None -> noop wrapper
    noop_factory = Population(lambda: nn.Linear(8, 4))
    assert not noop_factory.is_population
    assert noop_factory(x).shape == (5, 4)


def test_per_individual_inputs_and_nested_outputs():
    net = TwoBranchNet(8, 16, 4)
    pop = Population(net, pop_size = 4)

    x_pop = torch.randn(4, 6, 8)
    out, hidden = pop(x_pop, gain = 2.0, return_hidden = True, per_individual = True)
    assert out.shape == (4, 6, 4)
    assert hidden.shape == (4, 6, 16)

    for i in range(4):
        ind = pop[i]
        ref_out, ref_hidden = ind(x_pop[i], gain = 2.0, return_hidden = True)
        assert torch.allclose(out[i], ref_out, atol = 1e-5)
        assert torch.allclose(hidden[i], ref_hidden, atol = 1e-5)

    with pytest.raises(ValueError):
        pop(torch.randn(3, 6, 8), per_individual = True)


def test_ensemble_training_step_updates_all_members():
    net = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
    pop = Population(net, pop_size = 4)
    opt = torch.optim.Adam(pop.parameters(), lr = 1e-2)

    x = torch.randn(10, 8)
    y = torch.randn(10, 4)

    before = [p.detach().clone() for p in pop.parameters()]
    preds = pop(x)
    loss = (preds - y.unsqueeze(0)).pow(2).mean()
    loss.backward()
    opt.step()

    for old, new in zip(before, pop.parameters()):
        assert not torch.equal(old, new)


# 3. Indexing out individuals & using them as single networks again

def test_index_out_individuals_and_use_as_single_network():
    net = TwoBranchNet(8, 16, 4)
    pop = Population(net, pop_size = 4)
    x = torch.randn(5, 8)

    pop_out = pop(x)

    # integer, negative, and 0-d tensor indexing
    for idx in (0, 2, -1, torch.tensor(1)):
        ind = pop[idx]
        assert isinstance(ind, TwoBranchNet)

        # single-network forward matches population slice
        out_i = ind(x)
        assert out_i.shape == (5, 4)
        assert torch.allclose(out_i, pop_out[int(idx)], atol = 1e-5)

        # original custom methods & non-persistent buffers preserved
        assert ind.encode(x).shape == (5, 16)
        assert torch.equal(ind.temp, torch.tensor(0.5))

        # state_dict keys match original bare network exactly
        assert list(ind.state_dict().keys()) == list(net.state_dict().keys())

    # standalone training on indexed-out individual with standard torch.optim
    ind = pop[2]
    before_pop = pop.params_dict['head.weight'].detach().clone()
    opt = torch.optim.SGD(ind.parameters(), lr = 0.1)
    ind(x).pow(2).mean().backward()
    opt.step()

    # mutating extracted copy does not corrupt pop until explicitly written back
    assert torch.equal(pop.params_dict['head.weight'], before_pop)

    # write trained single network back into slot 2
    pop[2] = ind
    assert torch.allclose(pop(x)[2], ind(x), atol = 1e-5)
    assert torch.equal(pop.params_dict['head.weight'][0], before_pop[0])

    # re-wrap an indexed-out individual into a new Population
    pop2 = Population(ind, pop_size = 3)
    assert pop2(x).shape == (3, 5, 4)
    assert torch.allclose(pop2(x)[0], ind(x), atol = 1e-5)

    # iteration yields all individuals as single networks
    individuals = list(pop)
    assert len(individuals) == 4
    assert all(isinstance(m, TwoBranchNet) for m in individuals)


def test_slice_list_and_mask_indexing():
    net = nn.Linear(8, 4)
    pop = Population(net, pop_size = 5)
    x = torch.randn(6, 8)
    full_out = pop(x)

    # slice
    sub_slice = pop[1:4]
    assert isinstance(sub_slice, Population)
    assert len(sub_slice) == 3
    assert torch.allclose(sub_slice(x), full_out[1:4], atol = 1e-6)

    # list / resampling with duplicates
    sub_list = pop[[3, 0, 3]]
    assert len(sub_list) == 3
    assert torch.allclose(sub_list(x), full_out[[3, 0, 3]], atol = 1e-6)

    # boolean mask
    mask = torch.tensor([True, False, True, False, True])
    sub_mask = pop[mask]
    assert len(sub_mask) == 3
    assert torch.allclose(sub_mask(x), full_out[mask], atol = 1e-6)

    # slice assignment
    pop[0:2] = pop[3:5]
    assert torch.allclose(pop(x)[0:2], full_out[3:5], atol = 1e-6)


def test_live_view_gradients_flow_into_population():
    net = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
    pop = Population(net, pop_size = 4)
    x = torch.randn(6, 8)

    # single individual live view
    v1 = pop.view(1)
    assert isinstance(v1, PopulationView)
    assert v1.is_single
    assert len(v1) == 1

    out1 = v1(x)
    assert out1.shape == (6, 4)
    assert torch.allclose(out1, pop(x)[1], atol = 1e-5)

    out1.pow(2).mean().backward()
    w_grad = pop.grads_dict['0.weight']
    assert w_grad is not None
    assert w_grad[1].abs().sum() > 0
    assert torch.equal(w_grad[0], torch.zeros_like(w_grad[0]))
    assert torch.equal(w_grad[2], torch.zeros_like(w_grad[2]))
    assert torch.equal(w_grad[3], torch.zeros_like(w_grad[3]))

    # subset live view
    pop.zero_grad()
    v_sub = pop.view([0, 3])
    assert not v_sub.is_single
    assert len(v_sub) == 2
    out_sub = v_sub(x)
    assert out_sub.shape == (2, 6, 4)
    out_sub.pow(2).mean().backward()

    w_grad = pop.grads_dict['0.weight']
    assert w_grad[0].abs().sum() > 0
    assert w_grad[3].abs().sum() > 0
    assert torch.equal(w_grad[1], torch.zeros_like(w_grad[1]))
    assert torch.equal(w_grad[2], torch.zeros_like(w_grad[2]))


# 4. Edge cases: BatchNorm, Dropout, tied weights, frozen params, parameterless, state_dict, dtype

def test_batchnorm_dropout_and_train_eval_sync():
    net = nn.Sequential(
        nn.Linear(8, 16),
        nn.BatchNorm1d(16),
        nn.Dropout(0.5),
        nn.ReLU(),
        nn.Linear(16, 4),
    )
    pop = Population(net, pop_size = 3)
    x_pop = torch.randn(3, 8, 8) + torch.arange(3).view(3, 1, 1) * 5.0

    # train mode updates each member's running_mean independently
    pop.train()
    out_train = pop(x_pop, per_individual = True)
    assert out_train.shape == (3, 8, 4)
    out_train.sum().backward()

    rm = pop.buffers_dict['1.running_mean']
    assert not torch.allclose(rm[0], rm[1])

    # eval mode is deterministic and matches extracted individual
    pop.eval()
    x = torch.randn(4, 8)
    out_eval_1 = pop(x)
    out_eval_2 = pop(x)
    assert torch.equal(out_eval_1, out_eval_2)

    ind1 = pop[1]
    assert not ind1.training
    assert torch.allclose(ind1(x), out_eval_1[1], atol = 1e-5)


def test_tied_weights_and_frozen_parameters_preserved():
    net = TiedWeightNet(8)
    pop = Population(net, pop_size = 3)
    x = torch.randn(4, 8)

    assert len(pop.params_dict) == 1
    assert torch.allclose(pop(x)[0], net(x), atol = 1e-5)

    ind = pop[2]
    assert ind.a.weight is ind.b.weight
    assert torch.allclose(ind(x), pop(x)[2], atol = 1e-5)

    # frozen parameter preservation
    lin = nn.Linear(8, 4)
    lin.bias.requires_grad_(False)
    pop_f = Population(lin, pop_size = 3)
    assert pop_f.params_dict['weight'].requires_grad is True
    assert pop_f.params_dict['bias'].requires_grad is False
    ind_f = pop_f[1]
    assert ind_f.weight.requires_grad is True
    assert ind_f.bias.requires_grad is False


def test_parameterless_modules_identity_and_dropout():
    x = torch.randn(4, 16)

    pop_id = Population(nn.Identity(), pop_size = 3)
    out_id = pop_id(x)
    assert out_id.shape == (3, 4, 16)
    assert torch.equal(out_id[0], x)
    assert isinstance(pop_id[0], nn.Identity)

    pop_drop = Population(nn.Dropout(0.5), pop_size = 4)
    out_drop = pop_drop(x)
    assert out_drop.shape == (4, 4, 16)
    assert not torch.equal(out_drop[0], out_drop[1])


def test_state_dict_and_dtype_conversion():
    net1 = TwoBranchNet(8, 16, 4)
    net2 = TwoBranchNet(8, 16, 4)
    pop1 = Population(net1, pop_size = 3)
    pop2 = Population(net2, pop_size = 3)

    x = torch.randn(5, 8)
    assert not torch.allclose(pop1(x), pop2(x))

    pop2.load_state_dict(pop1.state_dict())
    assert torch.allclose(pop1(x), pop2(x), atol = 1e-6)

    # dtype conversion moves params and float buffers cleanly
    pop_f64 = pop1.double()
    x64 = x.double()
    out64 = pop_f64(x64)
    assert out64.dtype == torch.float64
    ind64 = pop_f64[0]
    assert ind64.head.weight.dtype == torch.float64
    assert torch.allclose(ind64(x64), out64[0], atol = 1e-10)


def test_nested_in_parent_module_and_noop_setitem():
    class Wrapper(Module):
        def __init__(self):
            super().__init__()
            self.policy = Population(
                nn.Sequential(nn.Linear(8, 16), nn.Dropout(0.5), nn.Linear(16, 4)),
                pop_size = 3
            )

        def forward(self, x):
            return self.policy(x)

    parent = Wrapper()
    x = torch.randn(4, 8)
    parent.eval()
    assert torch.equal(parent(x), parent(x))

    # noop __setitem__ copies in-place while preserving net identity
    net = nn.Linear(8, 4)
    other = nn.Linear(8, 4)
    noop = Population(net)
    noop[0] = other
    assert noop.unwrap() is net
    assert torch.allclose(noop(x), other(x))
    assert torch.allclose(noop.view(0)(x), other(x))

    # subset live view with per_individual=True
    pop = Population(nn.Linear(8, 4), pop_size = 4)
    x_sub = torch.randn(2, 5, 8)
    out_sub = pop.view([1, 3])(x_sub, per_individual = True)
    assert out_sub.shape == (2, 5, 4)
    assert torch.allclose(out_sub[0], pop[1](x_sub[0]), atol = 1e-5)
    assert torch.allclose(out_sub[1], pop[3](x_sub[1]), atol = 1e-5)

