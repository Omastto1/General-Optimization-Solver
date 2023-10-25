import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.operators.crossover.pntx import TwoPointCrossover

from pymoo.optimize import minimize
from pymoo.termination import get_termination

from rectangle import Rectangle, RectanglePenalty

from best_fit import squeeky_wheel_optimization_ga, visualize


import json
def parse_bkf_benchmark(file_path):
    asd = json.loads(open(file_path).read())

    W = asd['Objects'][0]['Length']
    items = [Rectangle(item['Length'], item["Height"]) for item in asd['Items']]

    return W, items

def parse_txt_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Extracting number of items and strip width
    n = int(lines[0].strip())
    W = int(lines[1].strip())

    items = []

    # Extracting items details
    for line in lines[2:]:
        index, width, height = map(int, line.strip().split())
        items.append(Rectangle(width, height))

    return W, items


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
        # print("running \n")
        # Calculate the total height based on the order in x
        rectangles = [RectanglePenalty(rectangle.width, rectangle.height, round(penalty)) for rectangle, penalty in zip(self.rectangles, x)]

        skyline, rectangles = squeeky_wheel_optimization_ga(rectangles, self.strip_width)

        total_height = max(skyline)

        out["F"] = total_height
        out["rectangles"] = rectangles

# strip_width, rectangles = parse_txt_file("data/2DSTRIPPACKING/zdf2.txt")
# strip_width = 130

results = {}


for i in range(12, 13):
    strip_width, rectangles = parse_bkf_benchmark(f"data/2DSTRIPPACKING/BKW/{i}.json")

    problem = StripPackingProblem(rectangles, strip_width)
    # termination = get_termination("n_eval", 300)

    algorithm = GA(
        pop_size=20,
        sampling=FloatRandomSampling(),
        crossover=TwoPointCrossover(prob=1.0),
        mutation=PolynomialMutation(),
        eliminate_duplicates=True
    )

    res = minimize(problem, algorithm, ("n_gen", 50), verbose=True)

    X = res.X
    F = res.F
    d = {}
    problem._evaluate(X, d)
    rectangles = d['rectangles']

    print("Best solution found: %s" % X)
    print("All solutions:")
    for x, f in zip(X, F):
        print("- x: %s, f: %s" % (x, f))

    results[f'BKW{i}'] = {"Height": F[0], "Rectangles": [rectangle.__dict__ for rectangle in rectangles]}

    # visualize(rectangles, strip_width, F[0])

print("asd")
with open("ga_results.json", 'a', encoding='utf-8') as file:
    json.dump(results, file)