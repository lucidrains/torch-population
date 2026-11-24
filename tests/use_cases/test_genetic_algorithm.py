import torch
from torch import nn

from torch_population import Population


def test_genetic_algorithm_tournament_crossover():
    torch.manual_seed(0)
    P, elites, tourney = 16, 4, 3
    x = torch.randn(64, 4)
    y = x @ torch.randn(4, 2)
    pop = Population(nn.Sequential(nn.Linear(4, 16), nn.Tanh(), nn.Linear(16, 2)), pop_size = P)

    fitness = lambda: -pop(x).sub(y).pow(2).mean(dim = (-2, -1))
    init_best = fitness().max().item()

    with torch.no_grad():
        for step in range(40):
            fit = fitness()
            losers = fit.topk(P - elites, largest = False).indices
            cand = torch.randint(P, (P - elites, 2, tourney))
            p1, p2 = cand.gather(-1, fit[cand].argmax(-1, keepdim = True)).squeeze(-1).unbind(-1)

            if step == 0:
                expected = 0.5 * (pop.flat_params[p1] + pop.flat_params[p2])

            pop[losers] = pop[p1].lerp_(pop.view(p2), 0.5)

            if step == 0:
                assert torch.allclose(pop.flat_params[losers], expected)

            pop.flat_params[losers] += 0.05 * torch.randn_like(pop.flat_params[losers])

    assert fitness().max().item() > init_best
