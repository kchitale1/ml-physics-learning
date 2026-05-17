"""
Day 7 — Regression project skeleton.

Target: fit f(x) = sin(5x) * exp(-x^2) with Gaussian noise, then study how
network capacity (width × depth) interacts with overfitting.

How to use this file:
    - Paste into a Jupyter notebook splitting on the `# %%` cell markers, OR
    - Run as a script: `python day7_regression_skeleton.py`
Fill in every `TODO`. Don't peek at PyTorch tutorials for the answers until
you've taken a real swing — the skeleton tells you WHAT each piece does;
your job is HOW.
"""

# %% ---------------------------------------------------------------- imports
# TODO: import torch, torch.nn as nn, torch.optim, DataLoader, Dataset,
# numpy as np, matplotlib.pyplot as plt. Also `from pathlib import Path` for
# checkpoint paths, and `import copy` if your Day 5 checkpointing uses
# deepcopy.
import torch
import torch.nn as nn  
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import copy
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %% --------------------------------------------------------- reproducibility
SEED = 0
# TODO: seed torch, numpy, and (if you have one) torch.cuda.manual_seed_all.
# Why: you'll be comparing configs head-to-head; eliminate run-to-run noise
# so capacity differences aren't drowned in seed variance.
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# %% --------------------------------------------------------- target function
def true_f(x):
    """f(x) = sin(5x) * exp(-x^2). Vectorized over a 1D tensor or array."""
    # TODO: 1 line.
    return torch.sin(5 * x) * torch.exp(-x ** 2)
    #raise NotImplementedError


# %% ----------------------------------------------------- synthetic dataset
# Design choices to make explicit:
#   - Domain: x in [-2, 2] is enough to see ~3 full oscillations modulated
#     by the Gaussian envelope. Don't go wider — the signal decays to ~0.
#   - N: 1000 points is plenty for a 1D problem. Don't overdo it.
#   - Noise: Gaussian, std = 0.1. Big enough that overfitting is visible,
#     small enough that the signal is still learnable.
#   - Splits: 70 / 15 / 15 train / val / test, randomly assigned (NOT by x
#     order — that would create a distribution-shift problem instead of an
#     overfitting one).

N = 1000
NOISE_STD = 0.1

# TODO: generate x ~ Uniform(-2, 2), shape (N, 1). Compute y_clean = true_f(x).
# Then y_noisy = y_clean + Normal(0, NOISE_STD). Keep everything as
# torch.float32 tensors with shape (N, 1).
x = torch.rand(N, 1) * 4 - 2  # Uniform(-2, 2)
y_clean = true_f(x)
y_noisy = y_clean + torch.randn_like(y_clean) * NOISE_STD

# TODO: shuffle indices and split into train / val / test. Store as three
# (x, y) tensor pairs.
indices = torch.randperm(N)
train_size = int(0.7 * N)
val_size = int(0.15 * N)
train_indices = indices[:train_size]
val_indices = indices[train_size:train_size + val_size]
test_indices = indices[train_size + val_size:]
x_train, y_train = x[train_indices], y_noisy[train_indices]
x_val, y_val = x[val_indices], y_noisy[val_indices]
x_test, y_test = x[test_indices], y_noisy[test_indices]

class RegressionDataset:  # TODO: inherit from torch.utils.data.Dataset
    """Wrap (x, y) tensors as a torch Dataset.
    Foundations Week, Day F is exactly this pattern."""

    def __init__(self, x, y):
        # TODO: store
        self.x = x
        self.y = y
        #raise NotImplementedError

    def __len__(self):
        # TODO
        return len(self.x)
        #raise NotImplementedError

    def __getitem__(self, idx):
        # TODO: return one (x_i, y_i) pair as tensors
        return self.x[idx], self.y[idx]
        #raise NotImplementedError


# TODO: build train_loader, val_loader, test_loader.
# Batch size: 64 is fine. shuffle=True for train, False for val/test.
train_dataset = RegressionDataset(x_train, y_train)
val_dataset = RegressionDataset(x_val, y_val)     
test_dataset = RegressionDataset(x_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# %% --------------------------------------------------- parametric MLP class
class MLP(nn.Module):  # TODO: inherit from nn.Module
    """A feedforward net with configurable hidden width and depth.

    width = neurons per hidden layer
    depth = number of HIDDEN layers (so the full layer count is depth + 2:
            input -> hidden_1 -> ... -> hidden_depth -> output)
    activation = nn module class, default nn.ReLU
    """

    def __init__(self, in_dim=1, out_dim=1, width=64, depth=2, activation=None):
        # TODO:
        #   1. super().__init__()
        #   2. Build a list of layers: Linear(in_dim, width), act,
        #      then (depth-1) repeats of [Linear(width, width), act],
        #      then Linear(width, out_dim) as the head (NO activation after).
        #   3. Wrap in nn.Sequential and store on self.
        # Hint: if depth == 1 you should have exactly one hidden layer.
        super().__init__()
        if activation is None:
            activation = nn.ReLU # default activation
        layers = []
        layers.append(nn.Linear(in_dim, width)) # input layer   
        layers.append(activation())             # activation after input layer
        for _ in range(depth - 1):
            layers.append(nn.Linear(width, width)) # hidden layers
            layers.append(activation())             # activation after hidden layers    
        layers.append(nn.Linear(width, out_dim)) # output layer
        self.net = nn.Sequential(*layers) # wrap in Sequential
        #raise NotImplementedError

    def forward(self, x):
        # TODO
        return self.net(x)
        #raise NotImplementedError


def count_params(model):
    """Number of trainable scalars — your honest measure of capacity."""
    # TODO: one-liner with sum(...) over model.parameters()
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    #raise NotImplementedError


# %% ----------------------------------------------- training (Day 5 + Day 6)
def train(
    model,
    train_loader,
    val_loader,
    device,
    epochs=500,
    lr=1e-3,
    patience=50,
    ckpt_path="best.pt",
):
    """Train one model. Reuse your Day 5/6 code as-is.
    Must return: dict with keys
        'train_loss': list[float], length = epochs actually run
        'val_loss':   list[float], same length
        'best_val':   float
        'best_epoch': int
    and load the best checkpoint back into `model` before returning.
    """
    # TODO: copy your Day 5/6 train() in here. The only thing this skeleton
    # cares about is the return contract above — the rest you already wrote.
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    patience_counter = 0
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        # TRAIN STEP
        model.train()
        epoch_train_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            pred = model(x_batch)
            loss = nn.MSELoss()(pred, y_batch)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() * x_batch.size(0)
        epoch_train_loss /= len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        # VAL STEP
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                pred = model(x_batch)
                loss = nn.MSELoss()(pred, y_batch)
                epoch_val_loss += loss.item() * x_batch.size(0)
        epoch_val_loss /= len(val_loader.dataset)
        val_losses.append(epoch_val_loss)

        # CHECKPOINTING
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), ckpt_path)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
        #print(f"Epoch {epoch+1}: train={epoch_train_loss:.4e} val={epoch_val_loss:.4e}")
    # Load best checkpoint before returning
    model.load_state_dict(torch.load(ckpt_path))
    best_epoch = len(train_losses) - patience_counter - 1

    return {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'best_val': best_val_loss,
        'best_epoch': best_epoch
    }
    #   raise NotImplementedError


# %% ----------------------------------------------------------- the sweep
# Four configs chosen to span the capacity axis.
# Predict before you run: which will underfit? overfit? hit the sweet spot?
# Write your prediction in a comment HERE before training. Compare after.
#
# Your prediction:
# TODO1
# tiny will underfit, small and medium will be just right and large will overfit. 
# The loss curves for tiny will show a high training and validation loss that doesn't improve much.
#  The residuals for tiny will show a clear pattern indicating underfitting. 
# For small and medium, the loss curves will show a steady decrease in training and validation loss, 
# with the validation loss reaching a low point before potentially increasing due to overfitting. 
# The residuals for small and medium will be more noise-like, indicating a good fit. 
# For large, the training loss will continue to decrease while the validation loss starts to increase after a certain point,
#  indicating overfitting. The residuals for large will show a structured pattern,
#  indicating that the model is capturing noise rather than the underlying signal.

CONFIGS = [
    # (name, width, depth)
    ("tiny",   8,   1),
    ("small",  32,  2),
    ("medium", 128, 3),
    ("large",  512, 4),
]

def run_sweep(configs, train_loader, val_loader, device):
    """Train every config, return a dict keyed by name with model + history."""
    results = {}
    for name, width, depth in configs:
        print(f"\n=== {name}: width={width}, depth={depth} ===")
        # TODO:
        #   1. Build model = MLP(width=width, depth=depth).to(device)
        #   2. Print count_params(model) — note it growing
        #   3. ckpt = f"ckpt_{name}.pt"
        #   4. history = train(model, train_loader, val_loader, device,
        #                      ckpt_path=ckpt)
        #   5. results[name] = {"model": model, "history": history,
        #                       "n_params": count_params(model)}
        model = MLP(width=width, depth=depth).to(device)
        print(f"Number of parameters: {count_params(model)}")
        ckpt = f"ckpt_{name}.pt"
        history = train(model, train_loader, val_loader, device, ckpt_path=ckpt)
        results[name] = {"model": model, "history": history, "n_params": count_params(model)} 

        #raise NotImplementedError

    return results

# Plot results: loss curves, predictions vs ground truth, residuals,
# and a summary of capacity vs val loss.

# %% --------------------------------------------------------------- plots
# One panel per config via subplots. No plt.show() inside loops, no overlaid
# everything-on-one-axes. Each function returns a figure you could save.

def _grid_shape(n):
    """Pick a (rows, cols) layout that fits n panels reasonably."""
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    return rows, cols


def _model_device(model):
    """Return the device the model's first parameter lives on."""
    return next(model.parameters()).device


def plot_losses(results):
    """One panel per config. Train (solid) + val (dashed) vs epoch, log-y.
    Vertical line marks best_epoch. Title shows n_params and best_val."""
    rows, cols = _grid_shape(len(results))
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows),
                             squeeze=False)
    for ax, (name, res) in zip(axes.flat, results.items()):
        h = res['history']
        epochs = range(1, len(h['train_loss']) + 1)
        ax.plot(epochs, h['train_loss'], label='train', linewidth=1.5)
        ax.plot(epochs, h['val_loss'], label='val',
                linestyle='--', linewidth=1.5)
        ax.axvline(h['best_epoch'] + 1, color='gray', linestyle=':',
                   label=f"best (epoch {h['best_epoch'] + 1})")
        ax.set_yscale('log')
        ax.set_xlabel('epoch')
        ax.set_ylabel('MSE loss')
        ax.set_title(f"{name}  ·  {res['n_params']:,} params  ·  "
                     f"best_val={h['best_val']:.4f}")
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, which='both', alpha=0.3)
    for ax in axes.flat[len(results):]:
        ax.set_visible(False)
    fig.suptitle('Training curves', fontsize=14, y=1.02)
    fig.tight_layout()

def plot_predictions(results, x_dense, y_true_dense, val_x, val_y):
    """One panel per config. Black line = true f(x). Red scatter = val data.
    Blue line = model prediction on x_dense."""
    rows, cols = _grid_shape(len(results))
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows),
                             squeeze=False)
    x_d_np = x_dense.detach().cpu().numpy().ravel()
    y_true_np = y_true_dense.detach().cpu().numpy().ravel()
    vx_np = val_x.detach().cpu().numpy().ravel()
    vy_np = val_y.detach().cpu().numpy().ravel()
    for ax, (name, res) in zip(axes.flat, results.items()):
        model = res['model']
        dev = _model_device(model)
        model.eval()
        with torch.no_grad():
            y_pred = model(x_dense.to(dev)).cpu().numpy().ravel()
        ax.plot(x_d_np, y_true_np, color='black', linewidth=2,
                label='true f(x)')
        ax.scatter(vx_np, vy_np, color='tab:red', alpha=0.4, s=20,
                   label='val data')
        ax.plot(x_d_np, y_pred, color='tab:blue', linewidth=1.8,
                label='prediction')
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.set_title(f"{name}  ·  {res['n_params']:,} params")
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
    for ax in axes.flat[len(results):]:
        ax.set_visible(False)
    fig.suptitle('Predictions vs ground truth', fontsize=14, y=1.02)
    fig.tight_layout()

def plot_residuals(results, val_x, val_y):
    """One panel per config. Scatter of (val_x, prediction - val_y) with
    a y=0 reference. Structured (non-noise) residuals = model isn't
    capturing signal in that region."""
    rows, cols = _grid_shape(len(results))
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows),
                             squeeze=False)
    vx_np = val_x.detach().cpu().numpy().ravel()
    vy_np = val_y.detach().cpu().numpy().ravel()
    for ax, (name, res) in zip(axes.flat, results.items()):
        model = res['model']
        dev = _model_device(model)
        model.eval()
        with torch.no_grad():
            y_pred = model(val_x.to(dev)).cpu().numpy().ravel()
        residuals = y_pred - vy_np
        ax.scatter(vx_np, residuals, alpha=0.5, s=20, color='tab:purple')
        ax.axhline(0, color='gray', linestyle='--')
        ax.set_xlabel('x')
        ax.set_ylabel('prediction - y_val')
        ax.set_title(f"{name}  ·  σ(resid)={residuals.std():.3f}")
        ax.grid(True, alpha=0.3)
    for ax in axes.flat[len(results):]:
        ax.set_visible(False)
    fig.suptitle('Residuals on validation set', fontsize=14, y=1.02)
    fig.tight_layout()

def plot_capacity_summary(results):
    """Single figure: x = n_params (log), y = best_val (log).
    This is the headline plot — it's the answer to 'capacity vs error.'"""
    names = list(results.keys())
    n_params = [results[n]['n_params'] for n in names]
    best_val = [results[n]['history']['best_val'] for n in names]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(n_params, best_val, marker='o', linewidth=1.5, markersize=10)
    for n, p, v in zip(names, n_params, best_val):
        ax.annotate(n, (p, v), textcoords='offset points',
                    xytext=(10, 5), fontsize=10)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('number of trainable parameters (log)')
    ax.set_ylabel('best validation loss (log)')
    ax.set_title('Capacity vs generalization')
    ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()

# %% --------------------------------------------------------------- main
if __name__ == "__main__":

    # TODO: train_loader, val_loader, test_loader already built above? Good.
    results = run_sweep(CONFIGS, train_loader, val_loader, device)

    # Dense grid for "predictions vs ground truth" plot
    # TODO: x_dense = linspace(-2, 2, 500), reshape (-1, 1), to(device)
    # TODO: y_true_dense = true_f(x_dense)
    x_dense = torch.linspace(-2, 2, 500).reshape(-1, 1).to(device)
    y_true_dense = true_f(x_dense)

    plot_losses(results)
    plot_predictions(results, x_dense, y_true_dense, x_val, y_val)
    plot_residuals(results, x_val, y_val)
    plot_capacity_summary(results)
    plt.show()

    # TODO: print a small table:
    #   name | n_params | best_val | best_epoch
    # This becomes the spine of your 1-page writeup.
    print(f"{'Name':<10} {'n_params':<10} {'best_val':<10} {'best_epoch':<10}")
    for name, res in results.items():
        print(f"{name:<10} {res['n_params']:<10} {res['history']['best_val']:<10.4e} {res['history']['best_epoch']:<10}")


# %% --------------------------------------------- reflection prompts (Day 7)
# After you've stared at the plots, answer each in 2-3 sentences in your
# learning log. These ARE the deliverable.
#
# 1. Which config underfit, and what does underfitting look like in
#    (a) the loss curves and (b) the residuals?
# 2. Which config overfit? Same question — how do you SEE it in both views?
# 3. Where was the sweet spot, and how did you know (not "lowest val loss" —
#    what did the predictions and residuals look like)?
# 4. Roughly what's the relationship between n_params and best_val loss?
#    Is it monotonic? Does it level off? Get worse?
# 5. What would you change about the experiment if you ran it again?