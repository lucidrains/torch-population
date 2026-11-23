import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from torch_population import Population

# config

POP_SIZE = 10
EPOCHS = 5
BATCH_SIZE = 256
LR = 3e-3

device = torch.device(
    'cuda' if torch.cuda.is_available()
    else 'mps' if torch.backends.mps.is_available()
    else 'cpu'
)

# data

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
full_train = datasets.MNIST('./data', train = True, download = True, transform = transform)
train_ds, val_ds = random_split(full_train, [55_000, 5_000], generator = torch.Generator().manual_seed(42))
test_ds = datasets.MNIST('./data', train = False, download = True, transform = transform)

train_loader = DataLoader(train_ds, batch_size = BATCH_SIZE, shuffle = True)
val_loader = DataLoader(val_ds, batch_size = 512)
test_loader = DataLoader(test_ds, batch_size = 512)

# population of 10 networks, each with its own AdamW optimizer state

net = nn.Sequential(nn.Flatten(), nn.Linear(784, 128), nn.SiLU(), nn.Linear(128, 10))
pop = Population(net, pop_size = POP_SIZE).to(device)
opt = pop.optimizer(torch.optim.AdamW, lr = LR, shared = False)

def per_member_ce(logits, y):
    targets = repeat(y, 'b -> p b', p = logits.shape[0])
    return F.cross_entropy(rearrange(logits, 'p b c -> p c b'), targets, reduction = 'none').mean(dim = -1)

@torch.no_grad()
def eval_pop(loader):
    pop.eval()
    loss_sum = torch.zeros(POP_SIZE, device = device)
    correct = torch.zeros(POP_SIZE, device = device)
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = pop(x)                                            # (p, b, 10)
        loss_sum += per_member_ce(logits, y) * y.size(0)
        correct += (logits.argmax(dim = -1) == y.unsqueeze(0)).sum(dim = -1)
        total += y.size(0)

    return loss_sum / total, correct / total

# train all 10 networks in parallel

for epoch in range(1, EPOCHS + 1):
    pop.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        per_member_ce(pop(x), y).sum().backward()
        opt.step()

    val_loss, val_acc = eval_pop(val_loader)
    print(
        f'epoch {epoch} | '
        f'val loss: min={val_loss.min():.4f} max={val_loss.max():.4f} mean={val_loss.mean():.4f} | '
        f'val acc: min={val_acc.min():.2%} max={val_acc.max():.2%} mean={val_acc.mean():.2%}'
    )

# verify independent per-member optimizer states & extract best individual by val loss

exp_avg = pop.population_parameters.opt_slot_flat('exp_avg')
assert exp_avg.shape == pop.population_parameters.flat_params.shape
assert not torch.allclose(exp_avg[0], exp_avg[1])

best_idx = val_loss.argmin().item()
best_net = pop[best_idx].eval()

with torch.no_grad():
    correct = sum((best_net(x.to(device)).argmax(dim = -1) == y.to(device)).sum().item() for x, y in test_loader)

print(f'\nbest member #{best_idx} (val loss {val_loss[best_idx]:.4f}, val acc {val_acc[best_idx]:.2%}) | test acc: {correct / len(test_ds):.2%}')
