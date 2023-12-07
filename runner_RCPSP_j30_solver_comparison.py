
import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 

import numpy as np
import networkx as nx

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

term = RobustTermination(SingleObjectiveSpaceTermination(tol = 0.1), period=30)
term_time = get_termination("time", "00:00:10")


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


naive_GA_solver_backward = RCPSPGASolver(algorithm, fitness_func_backward, term, seed=1, solver_name="naive GA backward_90_90_0.9_30")
naive_GA_solver_forward = RCPSPGASolver(algorithm, fitness_func_forward, term, seed=1, solver_name="naive GA forward_90_90_0.9_30")



naive_GA_solver_backward_time = RCPSPGASolver(algorithm, fitness_func_backward, term_time, seed=1, solver_name="naive GA backward_90_90_0.9_30_10sec")
naive_GA_solver_forward_time = RCPSPGASolver(algorithm, fitness_func_forward, term_time, seed=1, solver_name="naive GA forward_90_90_0.9_30_10sec")

naive_GA_solver_forward_time_paper = RCPSPGASolver(algorithm, bkrga_fitness_func, term_time, seed=1, solver_name="naive GA forward_90_90_0.9_30_10sec_paper")



### BRKGA


# values from https://pymoo.org/algorithms/soo/brkga.html 
algorithm = BRKGA(
    n_elites=15,
    n_offsprings=57,
    n_mutants=18,
    bias=0.7,
    eliminate_duplicates=MyElementwiseDuplicateElimination())

BRKGA_solver_forward = RCPSPGASolver(algorithm, fitness_func_forward, term, seed=1, solver_name="BRKGA_forward_15_57_18_0.7")
BRKGA_solver_backward = RCPSPGASolver(algorithm, fitness_func_backward, term, seed=1, solver_name="BRKGA_backward_15_57_18_0.7")

BRKGA_solver_forward_time = RCPSPGASolver(algorithm, fitness_func_forward, term_time, seed=1, solver_name="BRKGA_forward_15_57_18_0.7_10secaaaaa")
BRKGA_solver_backward_time = RCPSPGASolver(algorithm, fitness_func_backward, term_time, seed=1, solver_name="BRKGA_backward_15_57_18_0.7_10sec")


###

### RCPSP BRKGA


class MyElementwiseDuplicateElimination(ElementwiseDuplicateElimination):

    def is_equal(self, a, b):
        return (a.X.round(2) == b.X.round(2)).all()
    
algorithm = BRKGA(
    n_elites=5,
    n_offsprings=19,
    n_mutants=6,
    bias=0.7,
    eliminate_duplicates=MyElementwiseDuplicateElimination())

BRKGA_solver = PaperRCPSPGASolver(
    algorithm, bkrga_fitness_func, term, seed=1, solver_name="BRKGA_rcpsp_5_19_6_0.7")

BRKGA_solver_time = PaperRCPSPGASolver(
    algorithm, bkrga_fitness_func, term_time, seed=1, solver_name="BRKGA_rcpsp_5_19_6_0.7_10secaaaaapaper")

## CP

cp_solver15 = RCPSPCPSolver(TimeLimit=15, no_workers=1)
cp_solver10 = RCPSPCPSolver(TimeLimit=10, no_workers=1)



problem_type = "RCPSP"
benchmark_name = "j30.sm"

# SPECIFIC BENCHMARK INSTANCE
# benchmark = load_raw_benchmark(f"raw_data/{problem_type.lower()}/{benchmark_name}")
benchmark = load_benchmark(f"master_thesis_data/{problem_type}/{benchmark_name}")

# cp_solver15.solve(benchmark, validate=True, force_execution=True)
# BRKGA_solver_backward.solve(benchmark, validate=True, force_execution=True, force_dump=False)
# BRKGA_solver_forward.solve(benchmark, validate=True, force_execution=True, force_dump=False)
# naive_GA_solver_backward.solve(benchmark, validate=True, force_execution=True, force_dump=False)
# naive_GA_solver_forward.solve(benchmark, validate=True, force_execution=True, force_dump=False)

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

print(table_markdown)
print(table_markdown2)

# benchmark.dump(f"master_thesis_data/{problem_type}/{benchmark_name}")