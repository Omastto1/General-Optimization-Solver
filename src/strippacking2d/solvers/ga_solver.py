from docplex.cp.model import CpoModel
from collections import namedtuple
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize

from ...common.solver import GASolver
from ..utils.rectangle import Rectangle, RectanglePenalty


class StripPacking2DSolver(GASolver):
    def solve(self, algorithm, instance, fitness_func, termination, validate=False, visualize=False, force_execution=False):
        class StripPackingProblem(ElementwiseProblem):
            def __init__(self, rectangles, strip_width):
                super().__init__(n_var=len(rectangles),
                                n_obj=1,
                                n_constr=0,
                                xl=0,
                                xu=max(map(lambda x: x.width, rectangles)),
                                elementwise_evaluation=True)

                self.rectangles = rectangles
                self.strip_width = strip_width

            def _evaluate(self, x, out, *args, **kwargs):
                out = fitness_func(x, out)

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
        
        problem = StripPackingProblem(instance.rectangles, instance.strip_width)

        res = minimize(problem, algorithm, termination, verbose=True)

        if res:
            X = res.X
            total_height = res.F[0]
            d = {}
            problem._evaluate(X, d)
            rectangles = d['rectangles']

            placements = [rectangle.__dict__ for rectangle in rectangles]
            return total_height, placements, res
        else:
            return None, None, res

