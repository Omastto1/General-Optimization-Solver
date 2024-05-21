import random
import os
import sys

from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.algorithms.soo.nonconvex.brkga import BRKGA
from pymoo.algorithms.hyperparameters import HyperparameterProblem, MultiRun, stats_avg_nevals
from pymoo.algorithms.soo.nonconvex.g3pcx import G3PCX
from pymoo.core.mixed import MixedVariableGA
from pymoo.core.parameters import set_params, hierarchical
from pymoo.core.termination import TerminateIfAny
from pymoo.optimize import minimize
from pymoo.problems.single import Sphere
from pymoo.termination.fmin import MinimumFunctionValueTermination
from pymoo.termination.max_eval import MaximumFunctionCallTermination
from src.vrp.solvers.ga_model import *
from src.vrp.problem import *

path = sys.argv[1]

# randomly select a few instances

# instances = []
# for subfolder in os.listdir(path):
#     if os.path.isdir(os.path.join(path, subfolder)):
#         for file in os.listdir(os.path.join(path, subfolder)):
#             if file.endswith(".json"):
#                 instances.append(os.path.join(path, subfolder, file))
#
# # random.seed(0)
# train = random.sample(instances, 10)
# print(train)
# test = random.sample([instance for instance in instances if instance not in train], 20)
# print(test)

train = ['/solomon_100/R206.json', '/solomon_50/R205.json', '/solomon_100/C208.json', '/solomon_25/C201.json', '/solomon_100/R202.json', '/solomon_50/C206.json', '/solomon_50/C104.json', '/solomon_50/C109.json', '/solomon_50/RC207.json', '/solomon_25/RC102.json']
test = ['/solomon_25/C101.json', '/solomon_100/R109.json', '/solomon_50/R104.json', '/solomon_100/C108.json', '/solomon_25/RC201.json', '/solomon_50/C105.json', '/solomon_50/RC205.json', '/solomon_100/C101.json', '/solomon_50/C201.json', '/solomon_25/C208.json', '/solomon_25/C106.json', '/solomon_50/RC201.json', '/solomon_100/R111.json', '/solomon_25/R201.json', '/solomon_100/C106.json', '/solomon_100/C107.json', '/solomon_50/R207.json', '/solomon_100/C103.json', '/solomon_25/RC107.json', '/solomon_25/C103.json']

train = train[:5]

# algorithm = DE()
algorithm = BRKGA()


class VRPTW(ElementwiseProblem):
    """pymoo wrapper class
    """

    def __init__(self, instance, fitness_func_):
        super().__init__(n_var=instance.nb_customers, n_obj=1, xl=0, xu=1)
        self.instance = instance
        self.fitness_func = fitness_func_

    def _evaluate(self, x, out, *args, **kwargs):
        out = self.fitness_func(self.instance, x, out)

        assert "solution" not in out, "Do not use `solution` key, it is pymoo reserved keyword"

        return out


hyperparams = []

print(f"Training algorithm {algorithm.__class__.__name__}")

for instance in train:
    print(instance)
    instance = load_instance(path+instance)
    problem = VRPTW(instance, fitness_func)

    termination = TerminateIfAny(MinimumFunctionValueTermination(1e-5), MaximumFunctionCallTermination(500))

    performance = MultiRun(problem, seeds=[5, 50, 500], func_stats=stats_avg_nevals, termination=termination)

    res = minimize(HyperparameterProblem(algorithm, performance),
                   MixedVariableGA(pop_size=5),
                   ("time", "00:15:00"),
                   seed=1,
                   verbose=False)

    hyperparams.append(res.X)

print(hyperparams)

# evaluate on test instances

results = np.zeros((len(test), len(hyperparams) + 1))

for i, instance in enumerate(test):
    print(i)
    instance = load_instance(path+instance)

    # Add default hyperparameters
    # algorithm = DE()
    algorithm = BRKGA()
    problem = VRPTW(instance, fitness_func)
    res = minimize(problem, algorithm, termination=("time", "00:15:00"), seed=1, verbose=False)
    results[i, 0] = res.F[0].copy()

    for j, hyperparam in enumerate(hyperparams):
        set_params(algorithm, hierarchical(hyperparam))
        problem = VRPTW(instance, fitness_func)
        res = minimize(problem, algorithm, termination=("time", "00:15:00"), seed=1, verbose=False)
        results[i, j + 1] = res.F[0].copy()

# find the best hyperparameters

best_hyperparams = np.argmin(np.mean(results, axis=0))
print(f"Best hyperparameters: {best_hyperparams}/{len(hyperparams)+1}")
print(f"Validation results: {results.mean(axis=0)}")
print(hyperparams[best_hyperparams])
