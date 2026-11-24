import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from einops import rearrange, repeat

from torch_population import Population

# constants

POP_SIZE = 10
EPOCHS = 5
BATCH_SIZE = 256
EVAL_BATCH_SIZE = 512
LR = 3e-3
SEED = 42

device = torch.device(
    'cuda' if torch.cuda.is_available()
    else 'mps' if torch.backends.mps.is_available()
    else 'cpu'
)

# helpers

def per_member_ce(logits, targets):
    targets = repeat(targets, 'b -> p b', p = logits.shape[0])
    return F.cross_entropy(rearrange(logits, 'p b c -> p c b'), targets, reduction = 'none').mean(dim = -1)

@torch.no_grad()
def evaluate(pop, loader):
    pop.eval()
    pop_size = len(pop)
    loss_sum = torch.zeros(pop_size, device = device)
    member_correct = torch.zeros(pop_size, device = device)
    ensemble_correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = pop(x)  # (p, b, 10)

        loss_sum += per_member_ce(logits, y) * y.shape[0]
        member_correct += (logits.argmax(dim = -1) == rearrange(y, 'b -> 1 b')).sum(dim = -1)
        ensemble_correct += (logits.softmax(dim = -1).mean(dim = 0).argmax(dim = -1) == y).sum().item()
        total += y.shape[0]

    return loss_sum / total, member_correct / total, ensemble_correct / total

# main

def main():
    torch.manual_seed(SEED)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    full_train = datasets.MNIST('./data', train = True, download = True, transform = transform)
    train_ds, val_ds = random_split(full_train, [55_000, 5_000], generator = torch.Generator().manual_seed(SEED))
    test_ds = datasets.MNIST('./data', train = False, download = True, transform = transform)

    train_loader = DataLoader(train_ds, batch_size = BATCH_SIZE, shuffle = True)
    val_loader = DataLoader(val_ds, batch_size = EVAL_BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size = EVAL_BATCH_SIZE)

    # population of independent networks, each with its own colocated AdamW state

    net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 128),
        nn.SiLU(),
        nn.Linear(128, 10)
    )

    pop = Population(net, pop_size = POP_SIZE).to(device)
    opt = pop.optimizer(AdamW, lr = LR)

    for epoch in range(1, EPOCHS + 1):
        pop.train()

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            opt.zero_grad()
            per_member_ce(pop(x), y).sum().backward()
            opt.step()

        val_loss, val_acc, ens_acc = evaluate(pop, val_loader)

        print(
            f'epoch {epoch} | '
            f'val loss: min={val_loss.min():.4f} max={val_loss.max():.4f} mean={val_loss.mean():.4f} | '
            f'val acc: min={val_acc.min():.2%} max={val_acc.max():.2%} mean={val_acc.mean():.2%} | '
            f'ensemble: {ens_acc:.2%}'
        )

    # verify independent optimizer states, then evaluate best member and ensemble on test set

    exp_avg = pop.population_parameters.opt_slot_flat('exp_avg')
    assert exp_avg.shape == pop.population_parameters.flat_params.shape
    assert not torch.allclose(exp_avg[0], exp_avg[1])

    best_idx = val_loss.argmin().item()
    best_net = pop[best_idx].eval()

    with torch.no_grad():
        best_correct = sum((best_net(x.to(device)).argmax(dim = -1) == y.to(device)).sum().item() for x, y in test_loader)

    _, _, test_ens_acc = evaluate(pop, test_loader)

    print(
        f'\nbest member #{best_idx} (val loss {val_loss[best_idx]:.4f}, val acc {val_acc[best_idx]:.2%}) | '
        f'test acc: {best_correct / len(test_ds):.2%} | ensemble test acc: {test_ens_acc:.2%}'
    )

if __name__ == '__main__':
    main()
