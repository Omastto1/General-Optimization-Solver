import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize

from ...common.solver import GASolver


def indices_to_onehot(indices, num_classes):
    onehot = np.zeros((len(indices), num_classes))
    onehot[np.arange(len(indices)), indices] = 1
    return onehot


## python -m examples.example_BinPacking1D

class BinPacking1DGASolver(GASolver):
    def _solve(self, instance, validate=False, visualize=False, force_execution=False, update_history=True):
        class BinPackingProblem(ElementwiseProblem):
            def __init__(self, instance, fitness_func):
                super().__init__(n_var=len(instance.weights),
                                n_obj=1,
                                n_constr=1,  # One constraint: no bin overflow
                                xl=np.zeros(len(instance.weights)),
                                xu=np.ones(len(instance.weights)) * len(instance.weights) - 1,
                                elementwise_evaluation=True)
                self.instance = instance
                self.fitness_func = fitness_func

            def _evaluate(self, x, out, *args, **kwargs):
                out = self.fitness_func(self.instance, x, out)

                assert "solution" not in out, "Do not use `solution` key, it is pymoo reserved keyword"

                return out
        
        if not force_execution and len(instance._run_history) > 0:
            if instance.skip_on_optimal_solution():
                return None, None

        problem = BinPackingProblem(instance, self.fitness_func)
        res = minimize(problem, self.algorithm, self.termination, verbose=True, seed=self.seed,
                       callback=self.callback)

        if res.F is not None:
            X = np.floor(res.X).astype(int)
            fitness_value = res.F[0]

            d = {}
            problem._evaluate(X, d)
            placements = d['placements']

            assignment = indices_to_onehot(placements, len(instance.weights))

            if visualize:                
                instance.visualize(assignment)
        else:
            fitness_value = -1
            placements = []

        if update_history:
            solution_info = {"placements": placements}
            solution_progress = res.algorithm.callback.data['progress']
            self.add_run_to_history(instance, fitness_value, solution_info, solution_progress, exec_time=round(res.exec_time, 2))

        if res.F is not None:
            X = np.floor(res.X).astype(int)
            fitness_value = res.F[0]

            d = {}
            problem._evaluate(X, d)
            placements = {"item_bin_pos_assignment": d['placements']}
            return fitness_value, placements, res
        else:
            return None, None, res
        




