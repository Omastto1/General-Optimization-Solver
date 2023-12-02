from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize

from ...common.solver import GASolver
from ..utils.rectangle import Rectangle, RectanglePenalty


class StripPacking2DGASolver(GASolver):
    def _solve(self, instance, validate=False, visualize=False, force_execution=False):
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
                # print("running \n")
                # Calculate the total height based on the order in x
                rectangles = [RectanglePenalty(rectangle.width, rectangle.height, round(penalty)) for rectangle, penalty in zip(self.rectangles, x)]

                skyline, rectangles = squeeky_wheel_optimization_ga(rectangles, self.strip_width)

                total_height = max(skyline)

                out["F"] = total_height
                out["rectangles"] = rectangles

        if not force_execution and len(instance._run_history) > 0:
            if instance.skip_on_optimal_solution():
                return None, None
        
        problem = StripPackingProblem(instance, self.fitness_func)

        res = minimize(problem, self.algorithm, self.termination, verbose=True)

        if not res:
            return None, None, None
        
        X = res.X
        fitness_value = res.F[0]
        d = {}
        problem._evaluate(X, d)
        rectangles = d['rectangles']

        # rectangle..__dict__
        placements = [rectangle for rectangle in rectangles]

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
        
        self.add_run_to_history(instance, fitness_value, {"placements": placements}, exec_time=round(res.exec_time, 2))

        return fitness_value, placements, res

