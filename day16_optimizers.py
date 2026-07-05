"""
Day 16 — Optimizers deep dive
Empirical comparison of optimizers on the Day 7 regression problem:

    f(x) = sin(5x) * exp(-x^2)   with additive Gaussian noise

We train the SAME MLP (same architecture, same initialization) under each
optimizer, holding epochs and learning-rate-per-optimizer fixed enough to be
a fair-ish fight, and plot the training-loss curves on one log-scale axis.

Optimizers compared:
    - SGD (plain)
    - SGD + momentum
    - Adam
    - AdamW  (Adam with decoupled weight decay)
    - L-BFGS (bonus: second-order, full-batch, run as an Adam -> L-BFGS handoff)

Deliverable outputs:
    - loss_curves.png        : all optimizers, training loss vs epoch (log y)
    - fits.png               : each optimizer's learned function vs ground truth
    - console table          : final train loss + held-out val MSE per optimizer

Notes / choices:
    - Same seed re-initializes the network before every run so differences are
      due to the optimizer, not the starting weights.
    - The sin(5x) factor is high-frequency: watch how the optimizers fight to
      fit the wiggles (spectral bias, Day 12).
    - L-BFGS needs a full-batch, smooth loss and a closure(); it is handled in
      its own loop rather than shoehorned into the mini-batch loop.
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------------
SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------------------------
# 1. Data: f(x) = sin(5x) * exp(-x^2) + noise
# ----------------------------------------------------------------------------
def target_fn(x):
    return np.sin(5.0 * x) * np.exp(-(x ** 2))


def make_data(n=400, noise_std=0.05, x_range=(-3.0, 3.0)):
    x = np.random.uniform(x_range[0], x_range[1], size=(n, 1)).astype(np.float32)
    y_clean = target_fn(x).astype(np.float32)
    y = y_clean + np.random.normal(0.0, noise_std, size=y_clean.shape).astype(np.float32)
    return x, y


x_train_np, y_train_np = make_data(n=400, noise_std=0.05)
x_val_np, y_val_np = make_data(n=200, noise_std=0.05)

x_train = torch.from_numpy(x_train_np).to(DEVICE)
y_train = torch.from_numpy(y_train_np).to(DEVICE)
x_val = torch.from_numpy(x_val_np).to(DEVICE)
y_val = torch.from_numpy(y_val_np).to(DEVICE)


# ----------------------------------------------------------------------------
# 2. Model: a small MLP  (R -> R)
# ----------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, width=64, depth=3):
        super().__init__()
        layers = [nn.Linear(1, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# Build ONE reference model, snapshot its initial weights, and reuse that exact
# starting point for every optimizer so the comparison is apples-to-apples.
_reference = MLP().to(DEVICE)
INIT_STATE = copy.deepcopy(_reference.state_dict())


def fresh_model():
    m = MLP().to(DEVICE)
    m.load_state_dict(copy.deepcopy(INIT_STATE))
    return m


# ----------------------------------------------------------------------------
# 3. Training loops
# ----------------------------------------------------------------------------
loss_fn = nn.MSELoss()
EPOCHS = 400
BATCH = 64


def train_first_order(make_optimizer):
    """Mini-batch training loop for SGD/Adam-family optimizers.

    `make_optimizer` is a callable: params -> torch.optim.Optimizer
    Returns (loss_history, trained_model).
    """
    model = fresh_model()
    opt = make_optimizer(model.parameters())
    n = x_train.shape[0]
    history = []
    for _ in range(EPOCHS):
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = x_train[idx], y_train[idx]
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
        # record full-batch training loss at end of each epoch
        with torch.no_grad():
            history.append(loss_fn(model(x_train), y_train).item())
    return history, model


def train_lbfgs(adam_warmup_epochs=50, lbfgs_iters=100):
    """Adam warmup -> L-BFGS polish, both FULL-BATCH.

    This mirrors the canonical PINN recipe: Adam gets us into a good basin,
    then L-BFGS uses curvature to drive the loss much lower.
    """
    model = fresh_model()
    history = []

    # --- Stage 1: Adam warmup (full batch) ---
    adam = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(adam_warmup_epochs):
        adam.zero_grad()
        loss = loss_fn(model(x_train), y_train)
        loss.backward()
        adam.step()
        history.append(loss.item())

    # --- Stage 2: L-BFGS polish (full batch, closure-based) ---
    lbfgs = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=20,
        history_size=20,
        line_search_fn="strong_wolfe",
    )

    def closure():
        lbfgs.zero_grad()
        loss = loss_fn(model(x_train), y_train)
        loss.backward()
        return loss

    for _ in range(lbfgs_iters):
        loss = lbfgs.step(closure)
        history.append(float(loss))

    return history, model


# ----------------------------------------------------------------------------
# 4. Run the comparison
# ----------------------------------------------------------------------------
runs = {
    "SGD":            lambda: train_first_order(lambda p: torch.optim.SGD(p, lr=1e-2)),
    "SGD+momentum":   lambda: train_first_order(lambda p: torch.optim.SGD(p, lr=1e-2, momentum=0.9)),
    "Adam":           lambda: train_first_order(lambda p: torch.optim.Adam(p, lr=1e-3)),
    "AdamW":          lambda: train_first_order(lambda p: torch.optim.AdamW(p, lr=1e-3, weight_decay=1e-2)),
    "Adam->L-BFGS":   lambda: train_lbfgs(),
}

results = {}
for name, run in runs.items():
    torch.manual_seed(SEED)  # same data-shuffle stream per run
    history, model = run()
    with torch.no_grad():
        val_mse = loss_fn(model(x_val), y_val).item()
    results[name] = {"history": history, "model": model, "val_mse": val_mse}
    print(f"{name:14s}  final_train_loss={history[-1]:.5e}  val_mse={val_mse:.5e}")


# ----------------------------------------------------------------------------
# 5. Plots
# ----------------------------------------------------------------------------
# 5a. Training-loss curves (log y). x-axis = optimizer steps recorded.
plt.figure(figsize=(8, 5))
for name, r in results.items():
    plt.plot(r["history"], label=name)
plt.yscale("log")
plt.xlabel("recorded step (epoch, or iter for L-BFGS stage)")
plt.ylabel("training MSE (log scale)")
plt.title("Optimizer comparison — f(x)=sin(5x)exp(-x^2)")
plt.legend()
plt.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.savefig("loss_curves.png", dpi=130)
print("saved loss_curves.png")

# 5b. Learned fits vs ground truth
xs = np.linspace(-3, 3, 400).astype(np.float32).reshape(-1, 1)
xs_t = torch.from_numpy(xs).to(DEVICE)
plt.figure(figsize=(9, 5))
plt.scatter(x_val_np, y_val_np, s=8, alpha=0.3, color="gray", label="val data")
plt.plot(xs, target_fn(xs), "k--", lw=2, label="ground truth")
for name, r in results.items():
    with torch.no_grad():
        pred = r["model"](xs_t).cpu().numpy()
    plt.plot(xs, pred, lw=1.5, label=name)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Learned function per optimizer")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("fits.png", dpi=130)
print("saved fits.png")

# ----------------------------------------------------------------------------
# 6. For your learning log
# ----------------------------------------------------------------------------
print("\n--- for your learning log ---")
best = min(results.items(), key=lambda kv: kv[1]["val_mse"])
print(f"lowest val MSE: {best[0]} ({best[1]['val_mse']:.3e})")
print("Watch: which optimizer converged FAST early vs which reached the")
print("LOWEST final loss -- they are often not the same one.")