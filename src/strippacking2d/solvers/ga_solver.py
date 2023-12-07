from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize

from src.common.solver import GASolver


class StripPacking2DGASolver(GASolver):
    def _solve(self, instance, validate=False, visualize=False, force_execution=False, update_history=True):
        class StripPackingProblem(ElementwiseProblem):
            def __init__(self, instance, fitness_func):
                super().__init__(n_var=len(instance.rectangles),
                                n_obj=1,
                                n_constr=0,
                                xl=0,
                                xu=max(map(lambda x: x['width'], instance.rectangles)),
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
        
        problem = StripPackingProblem(instance, self.fitness_func)

        res = minimize(problem, self.algorithm, self.termination, verbose=True,
                       callback=self.callback)

        if not res:
            return None, None, None
        
        X = res.X
        fitness_value = res.F[0]
        d = {}
        problem._evaluate(X, d)
        placements = d['rectangles']

        if validate:
            try:
                print("Validating solution...")
                is_valid = instance.validate(None)
                if is_valid:
                    print("Solution is valid.")
                else:
                    print("Solution is invalid.")
            except AssertionError as e:
                print("Solution is invalid.")
                print(e)
                return None, None
        
        if visualize:
            instance.visualize(None, placements, fitness_value)

        if update_history:
            solution_progress = res.callback.data['progress']
            self.add_run_to_history(instance, fitness_value, {"placements": placements}, solution_progress, exec_time=round(res.exec_time, 2))

        return fitness_value, placements, res

