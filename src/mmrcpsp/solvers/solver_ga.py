import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from src.common.solver import GASolver


class MMRCPSPGASolver(GASolver):
    def _solve(self, instance, validate=False, visualize=False, force_execution=False, update_history=True):
        class MMRCPSP(ElementwiseProblem):

            def __init__(self, instance, fitness_func):
                # instance.no_jobs * 4 - 1 job priority, 3 mode priorities for each task
                super().__init__(n_var=instance.no_jobs * 4, n_obj=1,
                                 n_constr=instance.no_renewable_resources + instance.no_non_renewable_resources, xu=100, xl=0)
                print(
                    "WARNING: MMRCPSP GA SOLVER SUPPORTS ONLY 3 DIFFERENT MODES FOR EACH JOB")
                self.instance = instance
                self.fitness_func = fitness_func

            def _evaluate(self, x, out, *args, **kwargs):
                out = self.fitness_func(self.instance, x, out)

                assert "solution" not in out, "Do not use `solution` key, it is pymoo reserved keyword"

                return out

        if not force_execution and len(instance._run_history) > 0:
            if instance.skip_on_optimal_solution():
                return None, None

        problem = MMRCPSP(instance, self.fitness_func)
        res = minimize(problem, self.algorithm, self.termination, verbose=True, seed=self.seed,
                       callback=self.callback)

        if res.F is not None:
            X = res.X
            fitness_value = res.F[0]
            print('Objective value:', fitness_value)

            d = {}
            problem._evaluate(X, d)
            start_times = d['start_times']
            selected_modes = d['selected_modes']
            export = {"task_mode_assignment": [{"start": start_times[job], "end": start_times[job] + instance.durations[job][selected_mode],
                                                "mode": selected_mode, "name": f"Task_{job}_{selected_mode}"} for job, selected_mode in sorted(selected_modes.items())]}

            if validate:
                try:
                    print("Validating solution...")
                    instance.validate(export)
                    print("Solution is valid.")

                except AssertionError as e:
                    print("Solution is invalid.")
                    print(e)

                    self.add_run_to_history(
                        instance, fitness_value, start_times, [], is_valid=False)

                    return None, None, res

            if visualize:
                instance.visualize(export)
        else:
            fitness_value = -1
            start_times = []

        if update_history:
            solution_info = f"start_times: {start_times}"
            solution_progress = res.algorithm.callback.data['progress']
            self.add_run_to_history(instance, fitness_value, solution_info,
                                    solution_progress, exec_time=round(res.exec_time, 2))

        if res.F is not None:
            return fitness_value, start_times, res
        else:
            return None, None, res
