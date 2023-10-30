import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from src.common.solver import GASolver

class RCPSPGASolver(GASolver):
    def solve(self, algorithm, instance, fitness_func, termination, validate=False, visualize=False, force_execution=False):
        class RCPSP(ElementwiseProblem):
            
            def __init__(self, instance):
                super().__init__(n_var=instance.no_jobs, n_obj=1, n_constr=1, xu=100, xl=0)
                self.instance = instance

            def _evaluate(self, x, out, *args, **kwargs):
                out = fitness_func(self.instance, x, out)

                return out
        
        if not force_execution and len(instance._run_history) > 0:
            if instance.skip_on_optimal_solution():
                return None, None

        problem = RCPSP(instance)
        res = minimize(problem, algorithm, termination, verbose=True, seed=self.seed)

        if res.F is not None:
            X = np.floor(res.X).astype(int)
            fitness_value = res.F[0]
            print('Objective value:', fitness_value)

            d = {}
            problem._evaluate(X, d)
            start_times = d['start_times']

            if validate:
                try:
                    print("Validating solution...")
                    instance.validate(None, None, start_times)
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
                instance.visualize(None, None, start_times, [str(i) for i in range(instance.no_jobs)])
        else:
            fitness_value = -1
            start_times = []

        solution_info = f"start_times: {start_times}"
        self.add_run_to_history(instance, fitness_value, solution_info)

        if res.F is not None:
            return fitness_value, start_times, res
        else:
            return None, None, res
