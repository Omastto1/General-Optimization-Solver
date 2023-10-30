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
    def solve(self, algorithm, instance, fitness_func, termination, validate=False, visualize=False, force_execution=False):
        class BinPackingProblem(ElementwiseProblem):
            def __init__(self, weights, bin_capacity):
                super().__init__(n_var=len(weights),
                                n_obj=1,
                                n_constr=1,  # One constraint: no bin overflow
                                xl=np.zeros(len(weights)),
                                xu=np.ones(len(weights)) * len(weights) - 1,
                                elementwise_evaluation=True)
                self.weights = weights
                self.bin_capacity = bin_capacity

            def _evaluate(self, x, out, *args, **kwargs):
                out = fitness_func(self, x, out)
        
        if not force_execution and len(instance._run_history) > 0:
            if instance.skip_on_optimal_solution():
                return None, None

        problem = BinPackingProblem(instance.weights, instance.bin_capacity)
        res = minimize(problem, algorithm, termination, verbose=True, seed=self.seed)

        if res.F is not None:
            X = np.floor(res.X).astype(int)
            fitness_value = res.F[0]

            d = {}
            problem._evaluate(X, d)
            placements = d['placements']

            assignment = indices_to_onehot(placements, len(instance.weights))
            if validate:
                try:
                    print("Validating solution...")
                    instance.validate(instance, assignment, is_bin_used)
                    print("Solution is valid.")

                    # TODO
                    # obj_value = sol.get_objective_value()
                    # print("Project completion time:", obj_value)

                    # TODO
                    # instance.compare_to_reference(obj_value)
                except AssertionError as e:
                    print("Solution is invalid.")
                    print(e)

                    self.add_run_to_history(instance, fitness_value, start_times, is_valid=False)

                    return None, None, res

            if visualize:
                
                instance.visualize(assignment)
        else:
            fitness_value = -1
            placements = []

        solution_info = f"placements: {placements}"
        self.add_run_to_history(instance, fitness_value, solution_info)

        if res.F is not None:
            X = np.floor(res.X).astype(int)
            fitness_value = res.F[0]

            d = {}
            problem._evaluate(X, d)
            placements = d['placements']
            return fitness_value, placements, res
        else:
            return None, None, res
        




