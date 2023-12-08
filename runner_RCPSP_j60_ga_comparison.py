
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
from pymoo.termination import get_termination
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

from examples.brkga_2011_construct_schedules_forward import RCPSPGASolver as PaperRCPSPGASolver, fitness_func as bkrga_fitness_func

term_eval = get_termination("n_eval", 1000)


class MyElementwiseDuplicateElimination(ElementwiseDuplicateElimination):
    def is_equal(self, a, b):
        return (a.X.astype(int) == b.X.astype(int)).all()


algorithm = GA(
    pop_size=90,
    n_offsprings=90,
    sampling=FloatRandomSampling(),
    crossover=TwoPointCrossover(prob=0.9),
    mutation=PolynomialMutation(eta=30),
    selection=TournamentSelection(comp_by_cv_and_fitness),
    eliminate_duplicates=MyElementwiseDuplicateElimination()
)


naive_GA_solver_backward = RCPSPGASolver(algorithm, fitness_func_backward, term_eval, seed=1, solver_name="naive GA backward_90_90_0.9_30_1000evals")
naive_GA_solver_forward = RCPSPGASolver(algorithm, fitness_func_forward, term_eval, seed=1, solver_name="naive GA forward_90_90_0.9_30_1000evals")

naive_GA_solver_forward_paper = RCPSPGASolver(algorithm, bkrga_fitness_func, term_eval, seed=1, solver_name="naive GA forward_90_90_0.9_30_1000evals_paper")



### BRKGA


# values from https://pymoo.org/algorithms/soo/brkga.html 
algorithm = BRKGA(
    n_elites=15,
    n_offsprings=57,
    n_mutants=18,
    bias=0.7,
    eliminate_duplicates=MyElementwiseDuplicateElimination()
)

BRKGA_solver_forward = RCPSPGASolver(algorithm, fitness_func_forward, term_eval, seed=1, solver_name="BRKGA_forward_15_57_18_0.7_1000evals")


###

### RCPSP BRKGA
    
algorithm_small = BRKGA(
    n_elites=5,
    n_offsprings=19,
    n_mutants=6,
    bias=0.7,
    eliminate_duplicates=MyElementwiseDuplicateElimination()
)


BRKGA_solver_brkga = PaperRCPSPGASolver(
    algorithm, bkrga_fitness_func, term_eval, seed=1, solver_name="BRKGA_rcpsp_paper_5_19_6_0.7_1000evals")


import os

print("JOB ID")
print(os.environ['SLURM_ARRAY_JOB_ID'])
print("JOB ID")

problem_type = "RCPSP"
benchmark_name = "j60.sm"

# SPECIFIC BENCHMARK INSTANCE
benchmark = load_raw_benchmark(f"raw_data/{problem_type.lower()}/{benchmark_name}", no_instances=1)
# benchmark = load_benchmark(f"master_thesis_data/{problem_type}/{benchmark_name}")

# cp_solver15.solve(benchmark, validate=True, force_execution=True)
# BRKGA_solver_backward.solve(benchmark, validate=True, force_execution=True, force_dump=False)
# BRKGA_solver_forward.solve(benchmark, validate=True, force_execution=True, force_dump=False)
# naive_GA_solver_backward.solve(benchmark, validate=True, force_execution=True, force_dump=False)
naive_GA_solver_forward.solve(benchmark, validate=True, force_execution=True, force_dump=False)

# BRKGA_solver.solve(benchmark, visualize=False, validate=True, force_dump=False)

# cp_solver10.solve(benchmark, validate=True, force_execution=True)
# BRKGA_solver_backward_time.solve(benchmark, validate=True, force_execution=True, force_dump=False)
# BRKGA_solver_forward_time.solve(benchmark, validate=True, force_execution=True, force_dump=False)
# naive_GA_solver_backward_time.solve(benchmark, validate=True, force_execution=True, force_dump=False)
# naive_GA_solver_forward_time.solve(benchmark, validate=True, force_execution=True, force_dump=False)


# BRKGA_solver.solve(benchmark,  visualize=False, validate=True, force_execution=True, force_dump=False)
# BRKGA_solver_time.solve(benchmark, visualize=False, validate=True, force_execution=True, force_dump=False)
# naive_GA_solver_forward_time_paper.solve(benchmark, visualize=False, validate=True, force_execution=True, force_dump=False)


table_markdown = benchmark.generate_solver_comparison_markdown_table()
table_markdown2 = benchmark.generate_solver_comparison_percent_deviation_markdown_table()
table_markdown3 = benchmark.generate_solver_comparison_percent_deviation_markdown_table(compare_to_cplb=True)

print(table_markdown)
print(table_markdown2)
print(table_markdown3)

# benchmark.dump(f"master_thesis_data/{problem_type}/{benchmark_name}")