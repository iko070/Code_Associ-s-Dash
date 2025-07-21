import numpy as np
from pyDOE2 import lhs
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import pyswarms as ps
import itertools

# —— 1. Experimental design: Latin hypercube sampling —— #
def latin_hypercube_sampling(n_samples, bounds):
    dim = len(bounds)
    unit = lhs(dim, samples=n_samples)
    samples = np.zeros_like(unit)
    for i, (low, high) in enumerate(bounds):
        samples[:, i] = low + unit[:, i] * (high - low)
    return samples

# Define the range of process parameters: annealing temperature T, holding time t, cooling rate R
bounds = [(800, 950), (1440, 1680), (0.8, 1.2)]
X_train = latin_hypercube_sampling(20, bounds)

# —— 2. Synthesize experimental functions (can be replaced with real interfaces) —— #
def run_experiment(x):
    T, t, R = x
    Y = 90 \
        - ((T-925)/50)**2 * 5 \
        - ((t-1580)/100)**2 * 3 \
        - ((R-0.85)/0.1)**2 * 2 \
        + np.random.normal(0, 1)
    C = 5 + ((T-900)/100)**2 + np.random.normal(0, 0.1)
    return float(Y), float(C)

# Generate training data
Y_train, C_train = zip(*(run_experiment(x) for x in X_train))
Y_train, C_train = np.array(Y_train), np.array(C_train)

# —— 3. Kriging Agent Modeling —— #
kernel = Matern(nu=2.5)
gpr_y = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
gpr_c = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
gpr_y.fit(X_train, Y_train)
gpr_c.fit(X_train, C_train)

def surrogate_Y(x):
    return gpr_y.predict(x.reshape(1, -1))[0]

def surrogate_C(x):
    return gpr_c.predict(x.reshape(1, -1))[0]

# —— 4. Compound objective function: weighted approach —— #
wY, wC = 0.7, 0.3
def composite_obj_matrix(X):
    """
    X: np.ndarray, shape = (n_particles, 3)
    返回 shape = (n_particles,) 的目标值（越小越好）
    """
    preds_Y = gpr_y.predict(X)
    preds_C = gpr_c.predict(X)
    # Negative sign because pyswarms are minimized
    return -wY * preds_Y + wC * preds_C

# Boundary preparation
lb = np.array([b[0] for b in bounds])
ub = np.array([b[1] for b in bounds])
bounds_pso = (lb, ub)


# —— 5. PSO hyperparameter tuning —— #
c1_list = np.linspace(0.5, 2.0, 4)
c2_list = np.linspace(0.5, 2.0, 4)
w_list  = np.linspace(0.4, 0.9, 4)
n_list  = [20, 40, 60, 80, 100]

# hyperparameter space
param_grid = list(itertools.product(c1_list, c2_list, w_list, n_list))
results = []

for c1, c2, w, n in param_grid:
    options = {'c1': c1, 'c2': c2, 'w': w}
    optimizer = ps.single.GlobalBestPSO(
        n_particles=n,
        dimensions=3,
        options=options,
        bounds=bounds_pso
    )
    # Run 50 generations
    best_cost, best_pos = optimizer.optimize(composite_obj_matrix, iters=50, verbose=False)
    results.append({
        'c1': c1, 'c2': c2, 'w': w, 'n': n,
        'best_cost': best_cost
    })

# Finding the optimal hyperparameter configuration
best = min(results, key=lambda r: r['best_cost'])
print("Optimal PSO hyperparameters：", best)

# Check out the top five
top5 = sorted(results, key=lambda r: r['best_cost'])[:5]
print("\nTop 5 Configurations:")
for rank, r in enumerate(top5, 1):
    print(f"{rank}: c1={r['c1']}, c2={r['c2']}, w={r['w']}, n={r['n']}, cost={r['best_cost']:.3f}")