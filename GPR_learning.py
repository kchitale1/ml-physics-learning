import numpy as np

# Data
x_train = np.array([-1.0, 0.0, 1.0])
y_train = np.array([0.0, 1.0, 0.0])
x_star = 0.5

# Hyperparameters
ell = 1.0
sigma_f = 1.0
sigma_n = np.sqrt(0.1)

# RBF kernel
def rbf(x1, x2, ell=1.0, sigma_f=1.0):
    return sigma_f**2 * np.exp(-0.5 * (x1 - x2)**2 / ell**2)

# Build K
n = len(x_train)
K = np.array([[rbf(x_train[i], x_train[j]) for j in range(n)] for i in range(n)])
print("K =\n", np.round(K, 4))

# K + noise
K_noisy = K + sigma_n**2 * np.eye(n)

# K_star and K_starstar
K_star = np.array([rbf(xi, x_star) for xi in x_train])
K_starstar = rbf(x_star, x_star)
print("\nK_* =", np.round(K_star, 4))
print("K_** =", K_starstar)

# Posterior mean and variance
alpha = np.linalg.solve(K_noisy, y_train)
mu_star = K_star @ alpha
sigma2_star = K_starstar - K_star @ np.linalg.solve(K_noisy, K_star)

print(f"\nPosterior mean at x*=0.5:  {mu_star:.4f}")
print(f"Posterior variance at x*=0.5: {sigma2_star:.4f}")
print(f"Posterior std dev:            {np.sqrt(sigma2_star):.4f}")

import matplotlib.pyplot as plt

x_test = np.linspace(-2, 2, 200)
mu = []
sigma2 = []

for xs in x_test:
    ks = np.array([rbf(xi, xs) for xi in x_train])
    kss = rbf(xs, xs)
    mu.append(ks @ alpha)
    sigma2.append(kss - ks @ np.linalg.solve(K_noisy, ks))

mu = np.array(mu)
sigma = np.sqrt(np.array(sigma2))

plt.figure(figsize=(8, 4))
plt.plot(x_test, mu, 'b-', label='Posterior mean')
plt.fill_between(x_test, mu - 1.96*sigma, mu + 1.96*sigma, alpha=0.2, label='95% CI')
plt.scatter(x_train, y_train, color='black', zorder=5, label='Training data')
plt.axvline(0.5, color='red', linestyle='--', alpha=0.5, label='x*=0.5')
plt.legend(); plt.grid(True); plt.title('GP Posterior')
plt.show()