import torch
import gpytorch
import matplotlib.pyplot as plt


class SimpleGP(gpytorch.models.ExactGP):
    def __init__(self, x_train, y_train, likelihood):
        super().__init__(x_train, y_train, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)
    
# --- Data: noisy samples from sin(2πx)
torch.manual_seed(42)
x_train = torch.linspace(0, 1, 15)
y_train = torch.sin(2 * torch.pi * x_train) + 0.1 * torch.randn(15)

# --- Model setup
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = SimpleGP(x_train, y_train, likelihood)

# --- Training: maximize marginal log-likelihood
model.train()
likelihood.train()

optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

for i in range(100):
    optimizer.zero_grad()
    output = model(x_train)
    loss = -mll(output, y_train)   # minimize negative MLL
    loss.backward()
    optimizer.step()
    if (i+1) % 20 == 0:
        print(f"Iter {i+1}: loss={loss.item():.4f}  "
              f"ℓ={model.covar_module.base_kernel.lengthscale.item():.4f}  "
              f"σ_n={likelihood.noise.item():.4f}")

# --- Prediction
model.eval()
likelihood.eval()

x_test = torch.linspace(-0.1, 1.1, 200)
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    pred = likelihood(model(x_test))
    mean = pred.mean
    lower, upper = pred.confidence_region()  # ±2σ

for name, param in model.named_parameters():
    print(f"{name}: {param.item():.4f}")
    
# --- Plot
plt.figure(figsize=(8, 4))
plt.plot(x_test, mean, 'b-', label='Posterior mean')
plt.fill_between(x_test, lower, upper, alpha=0.2, label='95% CI')
plt.scatter(x_train, y_train, color='black', zorder=5, label='Training data')
plt.legend(); plt.grid(True)
plt.title('GPyTorch GP Regression')
plt.show()

# --- Print learned hyperparameters
print(f"\nLearned ℓ: {model.covar_module.base_kernel.lengthscale.item():.4f}")
print(f"Learned σ_f²: {model.covar_module.outputscale.item():.4f}")
print(f"Learned σ_n²: {likelihood.noise.item():.4f}")