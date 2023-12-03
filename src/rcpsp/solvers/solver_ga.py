import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from src.common.solver import GASolver

class RCPSPGASolver(GASolver):
    def _solve(self, instance, validate=False, visualize=False, force_execution=False, update_history=True):
        class RCPSP(ElementwiseProblem):
            
            def __init__(self, instance, fitness_func):
                super().__init__(n_var=instance.no_jobs, n_obj=1, n_constr=instance.no_renewable_resources, xu=100, xl=0)
                self.instance = instance
                self.fitness_func = fitness_func

            def _evaluate(self, x, out, *args, **kwargs):
                out = self.fitness_func(self.instance, x, out)

                assert "solution" not in out, "Do not use `solution` key, it is pymoo reserved keyword"

                return out
        
        if not force_execution and len(instance._run_history) > 0:
            if instance.skip_on_optimal_solution():
                return None, None

        problem = RCPSP(instance, self.fitness_func)
        res = minimize(problem, self.algorithm, self.termination, verbose=True, seed=self.seed)

        if res.F is not None:
            X = np.floor(res.X).astype(int)
            fitness_value = res.F[0]
            print('Objective value:', fitness_value)

            d = {}
            problem._evaluate(X, d)
            start_times = d['start_times']
            export = {"tasks_schedule": [{"start": start_times[i], "end": start_times[i] + instance.durations[i], "name": f"Task_{i}"} for i in range(instance.no_jobs)]}

            if validate:
                try:
                    print("Validating solution...")
                    instance.validate(None, None, export)
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
                instance.visualize(export)
        else:
            fitness_value = -1
            start_times = []

        if update_history:
            solution_info = f"start_times: {start_times}"
            self.add_run_to_history(instance, fitness_value, solution_info, exec_time=round(res.exec_time, 2))

        if res.F is not None:
            return fitness_value, start_times, res
        else:
            return None, None, res
