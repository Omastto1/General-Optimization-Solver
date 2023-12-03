
import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 


import numpy as np

from pymoo.algorithms.soo.nonconvex.ga import GA, comp_by_cv_and_fitness
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.core.duplicate import ElementwiseDuplicateElimination
from pymoo.termination.ftol import SingleObjectiveSpaceTermination
from pymoo.termination.robust import RobustTermination
from pymoo.operators.selection.tournament import TournamentSelection

from pymoo.algorithms.soo.nonconvex.brkga import BRKGA

from src.rcpsp.problem import RCPSP
from src.rcpsp.solvers.solver_cp import RCPSPCPSolver
from src.rcpsp.solvers.solver_ga import RCPSPGASolver
from src.general_optimization_solver import load_raw_instance, load_instance, load_raw_benchmark, load_benchmark


from ga_fitness_functions.rcpsp.naive_backward import fitness_func_backward
from ga_fitness_functions.rcpsp.naive_forward import fitness_func_forward


term = RobustTermination(SingleObjectiveSpaceTermination(tol = 0.1), period=20)


class MyElementwiseDuplicateElimination(ElementwiseDuplicateElimination):
    def is_equal(self, a, b):
        return (a.X.astype(int) == b.X.astype(int)).all()


algorithm = GA(
    pop_size=360,
    n_offsprings=360,
    sampling=FloatRandomSampling(),
    crossover=TwoPointCrossover(prob=0.9),
    mutation=PolynomialMutation(eta=30),
    selection=TournamentSelection(comp_by_cv_and_fitness),
    eliminate_duplicates=MyElementwiseDuplicateElimination()
)


naive_GA_solver_backward = RCPSPGASolver(algorithm, fitness_func_backward, term, seed=1, solver_name="naive GA backward_360_360_0.9_30")
naive_GA_solver_forward = RCPSPGASolver(algorithm, fitness_func_forward, term, seed=1, solver_name="naive GA forward_360_360_0.9_30")


### BRKGA


# values from https://pymoo.org/algorithms/soo/brkga.html 
algorithm = BRKGA(
    n_elites=60,
    n_offsprings=228,
    n_mutants=72,
    bias=0.7,
    eliminate_duplicates=MyElementwiseDuplicateElimination())

BRKGA_solver_forward = RCPSPGASolver(algorithm, fitness_func_forward, term, seed=1, solver_name="BRKGA_forward_60_228_72_0.7")
BRKGA_solver_backward = RCPSPGASolver(algorithm, fitness_func_backward, term, seed=1, solver_name="BRKGA_backward_60_228_72_0.7")

###

## CP

cp_solver = RCPSPCPSolver(TimeLimit=15, no_workers=1)



problem_type = "RCPSP"
benchmark_name = "j120.sm"

# SPECIFIC BENCHMARK INSTANCE
# benchmark = load_raw_benchmark(f"raw_data/{problem_type.lower()}/{benchmark_name}")
benchmark = load_benchmark(f"master_thesis_data/{problem_type}/{benchmark_name}")

# cp_solver.solve(benchmark, validate=True, force_execution=True, force_dump=False)
# BRKGA_solver_backward.solve(benchmark, validate=True, force_execution=True, force_dump=False)
BRKGA_solver_forward.solve(benchmark, validate=True, force_execution=True, force_dump=False)
# naive_GA_solver_backward.solve(benchmark, validate=True, force_execution=True, force_dump=False)
# naive_GA_solver_forward.solve(benchmark, validate=True, force_execution=True, force_dump=False)


table_markdown = benchmark.generate_solver_comparison_markdown_table()
table_markdown2 = benchmark.generate_solver_comparison_percent_deviation_markdown_table()

print(table_markdown)
print(table_markdown2)

benchmark.dump(f"master_thesis_data/{problem_type}/{benchmark_name}")