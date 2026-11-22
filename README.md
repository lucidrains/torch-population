## torch-population

Colocated population and ensemble wrapper for PyTorch. Parameters, buffers, optimizer states, gradient lanes, EMAs, and meta tensors all live together in one flat `PopulationParameters` container.

## Install

```bash
$ pip install torch-population
```

## Usage

### Simple: noop wrapper, population & indexing

```python
import torch
from torch import nn
from torch_population import Population

net = nn.Sequential(nn.Linear(16, 32), nn.SiLU(), nn.Linear(32, 4))
x = torch.randn(8, 16)

# zero-overhead noop wrapper (exact param aliasing)

pop = Population(net)
assert pop.unwrap() is net

# turn into a vectorized population (or call pop.populate(4))

pop = Population(net, pop_size = 4)

out = pop(x)                                              # (4, 8, 4) shared input
out = pop(torch.randn(4, 8, 16), per_individual = True)   # (4, 8, 4) per-individual input

# index out an individual as a standard single nn.Module and write it back

ind = pop[0]
assert ind(x).shape == (8, 4)
pop[0] = ind
```

### Colocated optimizers (`Adam`, `AdamW`, `SGD`, `Muon`, or custom `PopulationOptimizer`) & solo checkout

```python
from torch_population import Adam, SGD, Muon

opt = Adam(pop, lr = 1e-3, shared = False)   # separate (P, D) moments per individual
# opt = Adam(pop, lr = 1e-3, shared = True)  # shared (D,) moments across population (1/P memory)
# opt = Adam(pop, lr = 1e-3, shared = ('v',))
# opt = SGD(pop, lr = 1e-2, momentum = 0.9, nesterov = True)
# opt = Muon(pop, lr = 2e-2, momentum = 0.95) # batched Newton-Schulz on 2D+ matrices, AdamW on 1D

opt.zero_grad()
pop(x).pow(2).mean().backward()
opt.step()

# toggle sharing at runtime

opt.share()
opt.unshare()

# extract one individual + a native torch.optim.Adam, train solo, index both back in

ind, ind_opt = pop.extract(2)
ind_opt.zero_grad()
ind(x).pow(2).mean().backward()
ind_opt.step()
pop[2] = (ind, ind_opt)

# or with a context manager

with pop.checkout(2) as (ind, ind_opt):
    ind_opt.zero_grad()
    ind(x).pow(2).mean().backward()
    ind_opt.step()
```

### Unified `PopulationParameters`, PyTorch dicts & lifecycle

```python
pp = pop.population_parameters

pp.flat_params         # (4, D) contiguous parameters
pp.flat_exp_avg        # (4, D) or (D,) colocated first moment
pp.flat_exp_avg_sq     # (4, D) or (D,) colocated second moment

# single-individual PyTorch-compatible dict (zero-copy views; clone = True to copy)

ind_dict = pp[2]       # {'parameters', 'named_parameters', 'buffers', 'state_dict', 'optimizer_state', ...}
pop_dict = pp.as_dict()

# shared layers, subset forward, dynamic resizing & population soup

pop = Population(net, pop_size = 4, shared = ('0.bias',))
opt = pop.adamw(lr = 1e-3)

sub_out = pop(x, members = (0, 2))                        # (2, 8, 4)

pop.reset_(torch.tensor([True, False, False, False]))     # reinit masked params + buffers + opt state
pop.replace(3, 0, reset_opt = True)                       # copy member 0 into slot 3
pop.delete(1)                                             # P: 4 -> 3 in lockstep across params, opt, ema, meta
pop.append(0)                                             # P: 3 -> 4
avg_net = pop.mean()                                      # standalone nn.Module with averaged weights
```

### Advanced: differentiable meta-learning (DiscoRL) & colocated gradient lanes

```python
# 1. differentiable inner Adam unroll (one colocated PopulationParameters instead of separate params / ema / opt trees)

pop = Population(net, pop_size = 4)
opt = pop.adam(lr = 1e-3, shared = False)
pp = pop.population_parameters
ema_pp = pp.clone()

cur_pp = pp
for _ in range(2):
    inner_loss = pop(x, params = cur_pp).pow(2).mean()
    cur_pp = opt.functional_step(inner_loss, cur_pp, create_graph = True)

meta_loss = pop(x, params = cur_pp).pow(2).mean()
meta_loss.backward()                                      # gradients flow through inner Adam updates

pp.copy_from_(cur_pp.detach())
ema_pp.lerp_(pp, 0.05)

# 2. colocated gradient lanes & per-lane gradient EMAs

preds = pop(x)
g_task, = torch.autograd.grad(preds.pow(2).mean(), (pp.flat_params,), retain_graph = True)
g_reg,  = torch.autograd.grad(preds.abs().mean(),  (pp.flat_params,))

pp.set_lane('task', g_task)
pp.set_lane('reg',  g_reg)

ema_task = pp.update_lane_ema('task', decay = 0.9)
ema_reg  = pp.update_lane_ema('reg',  decay = 0.9)

dot = (ema_reg * ema_task).sum(dim = -1, keepdim = True)
denom = (ema_task * ema_task).sum(dim = -1, keepdim = True).clamp_min(1e-12)
pp.set_lane('combined', g_task + (ema_reg - (dot.clamp_max(0.) / denom) * ema_task))

opt.step(lane = 'combined')
```

## Tests

```bash
$ pytest
```
