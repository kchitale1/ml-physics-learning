"""
Day 8 — Initialization & Normalization experiment.

Same problem as Day 7: fit f(x) = sin(5x) * exp(-x^2) with Gaussian noise.
Same architecture: MLP, width=128, depth=3, ReLU activations.

Four training runs:
    bad_no_bn    — bad init (std=1.0),   no BatchNorm
    he_no_bn     — He/kaiming init,      no BatchNorm
    bad_with_bn  — bad init (std=1.0),   with BatchNorm
    he_with_bn   — He/kaiming init,      with BatchNorm

Everything else is identical: same data, same optimizer, same epochs.
That isolation is the experiment — only init and BN vary.

Deliverable: side-by-side loss curves + predictions showing
    1. How much init matters without BN
    2. Whether BN rescues bad init
    3. Whether BN helps further on top of good init

YOUR PREDICTION (fill this in before running):
# ...
"""

# %% ----------------------------------------------------------------- imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %% -------------------------------------------------------------- seed + data
# Fix everything so the 4 runs are comparable.
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

N         = 1000
NOISE_STD = 0.1

def true_f(x):
    return torch.sin(5 * x) * torch.exp(-x ** 2)

x       = torch.rand(N, 1) * 4 - 2          # Uniform(-2, 2)
y_clean = true_f(x)
y_noisy = y_clean + torch.randn_like(y_clean) * NOISE_STD

idx        = torch.randperm(N)
n_train    = int(0.70 * N)
n_val      = int(0.15 * N)
x_train, y_train = x[idx[:n_train]],              y_noisy[idx[:n_train]]
x_val,   y_val   = x[idx[n_train:n_train+n_val]], y_noisy[idx[n_train:n_train+n_val]]
x_test,  y_test  = x[idx[n_train+n_val:]],        y_noisy[idx[n_train+n_val:]]

class RegressionDataset(Dataset):
    def __init__(self, x, y): self.x, self.y = x, y
    def __len__(self):         return len(self.x)
    def __getitem__(self, i):  return self.x[i], self.y[i]

train_loader = DataLoader(RegressionDataset(x_train, y_train), batch_size=64, shuffle=True)
val_loader   = DataLoader(RegressionDataset(x_val,   y_val),   batch_size=64, shuffle=False)

# %% ---------------------------------------------------------------- MLP class
# Key change from Day 7: the `use_bn` flag inserts BatchNorm1d after each
# Linear layer, BEFORE the activation.  Order: Linear → BN → ReLU.
#
# Why before the activation?  BN normalises the raw linear combination — the
# pre-activation value.  If you placed BN after ReLU, half the inputs would
# already be clipped to zero and BN would be working on a skewed distribution.

class MLP(nn.Module):
    def __init__(self, in_dim=1, out_dim=1, width=128, depth=3, use_bn=False):
        super().__init__()
        layers = []

        # --- input → first hidden layer
        layers.append(nn.Linear(in_dim, width))
        if use_bn:
            layers.append(nn.BatchNorm1d(width))
        layers.append(nn.ReLU())

        # --- hidden → hidden (depth-1 more times)
        for _ in range(depth - 1):
            layers.append(nn.Linear(width, width))
            if use_bn:
                layers.append(nn.BatchNorm1d(width))
            layers.append(nn.ReLU())

        # --- final hidden → output  (no BN, no activation on the output)
        layers.append(nn.Linear(width, out_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# %% ------------------------------------------------------------- init helpers
# apply_init walks every Linear module in the model and replaces its weights.
#
# 'bad':  std=1.0  → variance = 1.0  → α = N * 1.0 = 128 per layer
#         After 3 layers: signal variance grows by 128^3 ≈ 2 million.
#
# 'he':   kaiming_normal_ → std = sqrt(2/N) → α ≈ 1  (stable for ReLU)
#
# Biases are zeroed in both cases — they don't affect variance propagation
# but zeroing them keeps the experiment clean.

def apply_init(model, init_type: str):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            if init_type == 'bad':
                nn.init.normal_(m.weight, mean=0.0, std=1.0)
                nn.init.zeros_(m.bias)
            elif init_type == 'he':
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)
            else:
                raise ValueError(f"Unknown init_type: {init_type!r}")


# %% -------------------------------------------------------------- train loop
# Identical to Day 7.  Returns history dict so we can plot afterwards.

def train(model, train_loader, val_loader, device,
          epochs=300, lr=1e-3, patience=40, ckpt_path="best.pt"):

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best_val   = float('inf')
    patience_counter = 0
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        # ---- train
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * xb.size(0)
        train_losses.append(running / len(train_loader.dataset))

        # ---- validate
        model.eval()
        running = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                running += criterion(model(xb), yb).item() * xb.size(0)
        vl = running / len(val_loader.dataset)
        val_losses.append(vl)

        # ---- checkpoint + early stop
        if vl < best_val:
            best_val = vl
            best_epoch = epoch
            torch.save(model.state_dict(), ckpt_path)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"  early stop at epoch {epoch+1}  best_val={best_val:.4e}")
            break

    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    return {
        'train_loss': train_losses,
        'val_loss':   val_losses,
        'best_val':   best_val,
        'best_epoch': best_epoch,
    }


# %% ----------------------------------------------------------- 4-config sweep
# Each config is (label, init_type, use_bn).
# We build a fresh model for each, apply the specified init, and train.

CONFIGS = [
    ("bad_no_bn",   "bad", False),
    ("he_no_bn",    "he",  False),
    ("bad_with_bn", "bad", True),
    ("he_with_bn",  "he",  True),
]

def run_sweep(configs, train_loader, val_loader, device):
    results = {}
    for name, init_type, use_bn in configs:
        print(f"\n=== {name} ===")
        model = MLP(use_bn=use_bn).to(device)
        apply_init(model, init_type)
        history = train(model, train_loader, val_loader, device,
                        ckpt_path=f"ckpt_day8_{name}.pt")
        results[name] = {"model": model, "history": history,
                         "init": init_type, "use_bn": use_bn}
        print(f"  best_val={history['best_val']:.4e}  "
              f"best_epoch={history['best_epoch']}")
    return results


# %% ------------------------------------------------------------------- plots

# Colour + style consistent across both plot types
STYLE = {
    "bad_no_bn":   dict(color="tab:red",    linestyle="--"),
    "he_no_bn":    dict(color="tab:blue",   linestyle="-"),
    "bad_with_bn": dict(color="tab:orange", linestyle="--"),
    "he_with_bn":  dict(color="tab:green",  linestyle="-"),
}

def plot_loss_curves(results):
    """
    All 4 val-loss curves on one axes (log-y) so convergence speed is directly
    comparable.  Also shows train loss as faint lines so you can spot
    train/val gaps (overfitting).
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    for name, res in results.items():
        h = res['history']
        epochs = range(1, len(h['val_loss']) + 1)
        s = STYLE[name]
        ax.plot(epochs, h['val_loss'],   label=name,           lw=2,   **s)
        ax.plot(epochs, h['train_loss'], alpha=0.25, lw=1,     **s)   # faint

    ax.set_yscale('log')
    ax.set_xlabel('epoch')
    ax.set_ylabel('MSE loss  (log scale)')
    ax.set_title('Day 8 — val loss: all 4 configs\n'
                 '(solid colour = val, faint = train)')
    ax.legend(loc='upper right')
    ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    return fig


def plot_predictions(results):
    """
    One panel per config.  True f(x) in black, val scatter in red, model
    prediction in the config's colour.  Lets you see if the model actually
    learned the function or got stuck.
    """
    x_dense     = torch.linspace(-2, 2, 500).reshape(-1, 1).to(device)
    y_true_np   = true_f(x_dense).cpu().numpy().ravel()
    x_dense_np  = x_dense.cpu().numpy().ravel()
    vx_np       = x_val.numpy().ravel()
    vy_np       = y_val.numpy().ravel()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    for ax, (name, res) in zip(axes.flat, results.items()):
        model = res['model'].eval()
        with torch.no_grad():
            y_pred = model(x_dense.to(device)).cpu().numpy().ravel()

        ax.plot(x_dense_np, y_true_np, color='black', lw=2,  label='true f(x)')
        ax.scatter(vx_np, vy_np, color='tab:red', alpha=0.3, s=15, label='val data')
        ax.plot(x_dense_np, y_pred,
                color=STYLE[name]['color'], lw=2, label='prediction')
        ax.set_title(f"{name}\nbest_val={res['history']['best_val']:.4e}")
        ax.set_xlabel('x')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Day 8 — predictions on validation set', fontsize=13, y=1.01)
    fig.tight_layout()
    return fig


# %% --------------------------------------------------------------------- main
if __name__ == "__main__":
    results = run_sweep(CONFIGS, train_loader, val_loader, device)

    fig1 = plot_loss_curves(results)
    fig2 = plot_predictions(results)
    plt.show()

    # Summary table
    print(f"\n{'config':<16}  {'init':<6}  {'BN':<5}  {'best_val':>10}  {'best_epoch':>10}")
    print("-" * 56)
    for name, res in results.items():
        h = res['history']
        print(f"{name:<16}  {res['init']:<6}  {str(res['use_bn']):<5}  "
              f"{h['best_val']:>10.4e}  {h['best_epoch']:>10}")

# %% ---------------------------------------------------------------- questions
# After you've stared at the plots, answer these in your learning log:
#
# 1. In plot_loss_curves: how many epochs did bad_no_bn need before it
#    started making real progress (if it ever did)?  Why?
# almost 25 epochs and the progress was very slow. This is because the bad init starts
# the model in a non-optimal region of the loss landscape, where the signal variance is huge
#  and gradients are unstable. It takes a long time for the model to "escape" 
# this region and start learning effectively.
#
# 2. Compare bad_no_bn vs bad_with_bn.  What did BatchNorm visibly do to
#    the bad-init curve?  Does this match your variance-propagation intuition?
# BatchNorm significantly improved the convergence of the bad init model. The 
# bad_no_bn curve shows a slow and erratic decrease in loss, while the bad_with_bn 
# curve shows a much smoother and faster convergence. This matches the 
# variance-propagation intuition because BatchNorm normalizes the activations, 
# which helps to stabilize the training process and allows the model to learn 
# effectively even with a bad initialization.
#
# 3. Compare he_no_bn vs he_with_bn.  Does BatchNorm help further on top
#    of already-good init, or is the gain marginal?
# The gain from BatchNorm on top of the already good He initialization is marginal.
#
# 4. Look at plot_predictions for bad_no_bn.  Did the model learn the
#    function at all, or is it a flat/random prediction?  What does that
#    tell you about what the weights are doing?
# The bad_no_bn model did not learn the function at all; 
# its predictions are essentially flat and random. This indicates that the weights 
# in the bad_no_bn model are not effectively capturing the underlying structure of 
# the data, likely due to the poor initialization leading to unstable gradients and 
# ineffective learning.
#
# 5. In one sentence: what is the practical takeaway for your future models?
# Definitely need to use a good initialization strategy (like He for ReLU) and BatchNorm 
# can be a powerful tool to rescue bad initializations and improve training stability,
#  but it may not provide significant benefits on top of already good initializations but 
# should be used for deeper models.
# %%
