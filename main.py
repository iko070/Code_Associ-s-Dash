import numpy as np
from pyDOE2 import lhs
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from deap import base, creator, tools, algorithms
import pyswarms as ps

# 1. Experimental design: Latin hypercube sampling
def latin_hypercube_sampling(n_samples, bounds):
    dim = len(bounds)
    unit = lhs(dim, samples=n_samples)
    samples = np.zeros_like(unit)
    for i, (low, high) in enumerate(bounds):
        samples[:, i] = low + unit[:, i] * (high - low)
    return samples

bounds = [(800, 950), (1440, 1680), (0.8, 1.2)]
X_train = latin_hypercube_sampling(20, bounds)

# # 2. Real or simulated experimental interface: return (yield, cycle_time)
# def run_experiment(x):
#     # TODO: 
#     Y = ...  
#     C = ...
#     return Y, C

# Y_train, C_train = zip(*(run_experiment(x) for x in X_train))
# Y_train, C_train = np.array(Y_train), np.array(C_train)

# 2. Synthetic experimental functions
def run_experiment(x):
    T, t, R = x
    # Synthesize yield function: optimize around (925,1580,0.85), add noise
    Y = 90 \
        - ((T-925)/50)**2 * 5 \
        - ((t-1580)/100)**2 * 3 \
        - ((R-0.85)/0.1)**2 * 2 \
        + np.random.normal(0, 1)
    
    C = 5 + ((T-900)/100)**2 + np.random.normal(0, 0.1)
    return float(Y), float(C)

# Getting Training Data
Y_train, C_train = zip(*(run_experiment(x) for x in X_train))
Y_train, C_train = np.array(Y_train), np.array(C_train)

# 3. Kriging model (GP)
kernel = Matern(nu=2.5)
gpr_y = GaussianProcessRegressor(kernel=kernel, normalize_y=True).fit(X_train, Y_train)
gpr_c = GaussianProcessRegressor(kernel=kernel, normalize_y=True).fit(X_train, C_train)

def surrogate_Y(x):
    return gpr_y.predict(x.reshape(1,-1))[0]
def surrogate_C(x):
    return gpr_c.predict(x.reshape(1,-1))[0]

# 4. Multi-objective NSGA-II: Generating the Pareto Frontier
creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()
for i, (low, high) in enumerate(bounds):
    toolbox.register(f"attr_{i}", np.random.uniform, low, high)
toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_0, toolbox.attr_1, toolbox.attr_2), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def eval_multi(ind):
    x = np.array(ind)
    return surrogate_Y(x), surrogate_C(x)

toolbox.register("evaluate", eval_multi)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=[10,50,0.05], indpb=0.2)
toolbox.register("select", tools.selNSGA2)

pop = toolbox.population(n=100)
algorithms.eaMuPlusLambda(pop, toolbox, mu=100, lambda_=100, cxpb=0.6, mutpb=0.3,
                          ngen=50, stats=None, halloffame=None, verbose=False)

pareto_front = tools.sortNondominated(pop, k=len(pop), first_front_only=True)[0]

# 5. pyswarms PSO refined search
# Synthetic targets (weighting method)
wY, wC = 0.7, 0.3
def composite_obj_matrix(X):
    
    n, _ = X.shape
    vals = np.zeros(n)
    for i in range(n):
        xi = X[i]
        vals[i] = -wY * surrogate_Y(xi) + wC * surrogate_C(xi)
    return vals

# Boundary
lb = np.array([b[0] for b in bounds])
ub = np.array([b[1] for b in bounds])
bounds_pso = (lb, ub)

# Global best optimal PSO
options = {'c1': 1.0, 'c2': 0.5, 'w': 0.4}
optimizer = ps.single.GlobalBestPSO(n_particles=60, dimensions=3,
                                   options=options, bounds=bounds_pso)

# run optimization
best_cost, best_pos = optimizer.optimize(composite_obj_matrix, iters=100)

print("Optimal parameters (pyswarms):", best_pos)
print("Predicted Yield:", surrogate_Y(best_pos),
      "Cycle Time:", surrogate_C(best_pos))

# 6. 现场验证
Y_val, C_val = run_experiment(best_pos)
print(f"Validation → Yield: {Y_val:.1f}%, Cycle Time: {C_val:.2f}h")
