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

import os

problem_type = "MMRCPSP"
benchmark_name = "j120.sm"


id = int(os.environ['SLURM_ARRAY_TASK_ID'])

print("JOB ID")
print(id)

parameter = id // 5 + 1
instance = id % 5

# ranges from j601_1.sm to j60_5.sm
if id % 5 == 0:
    parameter -= 1
    instance = 5


instance = load_raw_instance(
    "raw_data/mm-rcpsp/MMLIB50/J50{parameter}_{instance}.mm", "", "c15")


# cp_solution, cp_variables, sol = 
MMRCPSPCPSolver(TimeLimit=5).solve(instance, validate=True, visualize=False, force_execution=True, force_dump=False)

MMRCPSPGASolver(algorithm, fitness_func_forward, (
    "n_evals", 1000)).solve(instance, validate=True, visualize=False, force_execution=True, force_dump=False)

# table = benchmark.generate_solver_comparison_percent_deviation_markdown_table(compare_to_cplb=True)

# print(table)

instance.dump()