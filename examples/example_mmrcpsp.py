import numpy as np

from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.soo.nonconvex.brkga import BRKGA
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.core.duplicate import ElementwiseDuplicateElimination

from src.general_optimization_solver import load_raw_instance, load_instance, load_raw_benchmark
from src.mmrcpsp.solvers.solver_ga import MMRCPSPGASolver
from src.mmrcpsp.solvers.solver_cp import MMRCPSPCPSolver


from ga_fitness_functions.mmrcpsp.naive_forward import fitness_func_forward

# python -m examples.example_rcpsp


# Define the algorithm
algorithm = GA(
    pop_size=20,
    n_offsprings=50,
    sampling=FloatRandomSampling(),
    crossover=TwoPointCrossover(prob=0.9),
    mutation=PolynomialMutation(eta=3),
    eliminate_duplicates=True
)


class MyElementwiseDuplicateElimination(ElementwiseDuplicateElimination):

    def is_equal(self, a, b):
        return (a.X.round(2) == b.X.round(2)).all()


algorithm = BRKGA(
    n_elites=15,
    n_offsprings=70,
    n_mutants=15,
    bias=0.7,
    eliminate_duplicates=MyElementwiseDuplicateElimination()
)

# BENCHMARK TEST
# benchmark = load_raw_benchmark("raw_data/mm-rcpsp/c15.mm", "raw_data/mm-rcpsp/c15opt.mm.html", no_instances=2, force_dump=False)

# MMRCPSPCPSolver(TimeLimit=3).solve(benchmark, validate=True, visualize=True, force_execution=True)
# MMRCPSPGASolver(algorithm, fitness_func_forward, ("n_gen", 10)).solve(benchmark, validate=True, visualize=True, force_execution=True)

# table_markdown = benchmark.generate_solver_comparison_markdown_table()
# print(table_markdown)


# BENCHMARK TEST END


# instance = load_raw_instance(
#     "raw_data/mm-rcpsp/c15.mm/c1510_1.mm", "raw_data/mm-rcpsp/c15opt.mm.html", "c15")
# instance = load_instance("data/RCPSP/CV/cv1.json")
benchmark = load_raw_benchmark(
    "raw_data/mm-rcpsp/c15.mm", "raw_data/mm-rcpsp/c15opt.mm.html", "c15", no_instances=2)


# cp_solution, cp_variables, sol = 
MMRCPSPCPSolver(TimeLimit=5).solve(benchmark, validate=True, visualize=False, force_execution=True, force_dump=False)


# ga_fitness_value, ga_startimes, ga_solution = 
MMRCPSPGASolver(algorithm, fitness_func_forward, (
    "n_evals", 1000)).solve(benchmark, validate=True, visualize=False, force_execution=True, force_dump=False)
# print("Best solution found: \nX = ", ga_solution.X)

table = benchmark.generate_solver_comparison_percent_deviation_markdown_table(compare_to_cplb=True)

print(table)

benchmark.dump()