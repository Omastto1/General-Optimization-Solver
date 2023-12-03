import json

from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.pntx import TwoPointCrossover


from src.strippacking2d.solvers.ga_solver import StripPacking2DGASolver

from src.general_optimization_solver import load_raw_instance
from ga_fitness_functions.strip_packing_2d.best_fit_ga.best_fit import fitness_func

results = {}


algorithm = GA(
    pop_size=20,
    sampling=FloatRandomSampling(),
    crossover=TwoPointCrossover(prob=1.0),
    mutation=PolynomialMutation(),
    eliminate_duplicates=True
)


from pymoo.termination.ftol import SingleObjectiveSpaceTermination
from pymoo.termination.robust import RobustTermination
term = RobustTermination(SingleObjectiveSpaceTermination(tol = 0.1), period=30)
for i in range(12, 13):
    instance = load_raw_instance(f"raw_data/2d_strip_packing/BKW/{i}.json", "", "bkw", )

    # fitness_value, placements, res = StripPacking2DGASolver(algorithm, fitness_func, ("n_gen", 20)).solve(instance, visualize=False)
    fitness_value, placements, res = StripPacking2DGASolver(algorithm, fitness_func, term).solve(instance, visualize=False)

#     res = minimize(problem, algorithm, ("n_gen", 2), verbose=True)

    # X = res.X
    # F = res.F
    # d = {}
    # rectangles = d['rectangles']

    # print("Best solution found: %s" % X)
    # print("All solutions:")
    # for x, f in zip(X, F):
    #     print("- x: %s, f: %s" % (x, f))

    # results[f'BKW{i}'] = {"Height": F[0], "Rectangles": [rectangle.__dict__ for rectangle in rectangles]}

    # visualize(placements, instance.strip_width, fitness_value)

# print("asd")
# with open("ga_results.json", 'a', encoding='utf-8') as file:
#     json.dump(results, file)