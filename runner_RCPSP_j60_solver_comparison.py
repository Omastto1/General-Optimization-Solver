
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


term = RobustTermination(SingleObjectiveSpaceTermination(tol = 0.1), period=20)
term_time = get_termination("time", "00:00:30")


class MyElementwiseDuplicateElimination(ElementwiseDuplicateElimination):
    def is_equal(self, a, b):
        return (a.X.astype(int) == b.X.astype(int)).all()


algorithm = GA(
    pop_size=180,
    n_offsprings=180,
    sampling=FloatRandomSampling(),
    crossover=TwoPointCrossover(prob=0.9),
    mutation=PolynomialMutation(eta=30),
    selection=TournamentSelection(comp_by_cv_and_fitness),
    eliminate_duplicates=MyElementwiseDuplicateElimination()
)


naive_GA_solver_backward = RCPSPGASolver(algorithm, fitness_func_backward, term, seed=1, solver_name="naive GA backward_180_180_0.9_30")
naive_GA_solver_forward = RCPSPGASolver(algorithm, fitness_func_forward, term, seed=1, solver_name="naive GA forward_180_180_0.9_30")

from examples.brkga_2011_construct_schedules_forward import RCPSPGASolver as PaperRCPSPGASolver, fitness_func as bkrga_fitness_func
naive_GA_solver_forward_time_paper = RCPSPGASolver(algorithm, bkrga_fitness_func, term_time, seed=1, solver_name="naive GA forward_180_180_0.9_30_30sec_paper")

algorithm = GA(
    pop_size=30,
    n_offsprings=30,
    sampling=FloatRandomSampling(),
    crossover=TwoPointCrossover(prob=0.9),
    mutation=PolynomialMutation(eta=30),
    selection=TournamentSelection(comp_by_cv_and_fitness),
    eliminate_duplicates=MyElementwiseDuplicateElimination()
)


naive_GA_solver_backward_small = RCPSPGASolver(algorithm, fitness_func_backward, term_time, seed=1, solver_name="naive GA backward_30_30_0.9_30_60sec")
naive_GA_solver_forward_small = RCPSPGASolver(algorithm, fitness_func_forward, term_time, seed=1, solver_name="naive GA forward_30_30_0.9_30_60sec")



### BRKGA


# values from https://pymoo.org/algorithms/soo/brkga.html 
algorithm = BRKGA(
    n_elites=30,
    n_offsprings=114,
    n_mutants=36,
    bias=0.7,
    eliminate_duplicates=MyElementwiseDuplicateElimination())

BRKGA_solver_forward = RCPSPGASolver(algorithm, fitness_func_forward, term, seed=1, solver_name="BRKGA_forwardaaa_30_114_36_0.7")
BRKGA_solver_backward = RCPSPGASolver(algorithm, fitness_func_backward, term, seed=1, solver_name="BRKGA_backwardaa_30_114_36_0.7")

algorithm = BRKGA(
    n_elites=5,
    n_offsprings=19,
    n_mutants=6,
    bias=0.7,
    eliminate_duplicates=MyElementwiseDuplicateElimination())

BRKGA_solver_forward_small = RCPSPGASolver(algorithm, fitness_func_forward, term_time, seed=1, solver_name="BRKGA_forwarda_5_19_6_0.7_60sec")
BRKGA_solver_backward_small = RCPSPGASolver(algorithm, fitness_func_backward, term_time, seed=1, solver_name="BRKGA_backward_5_19_6_0.7_60sec")

###

## CP

cp_solver = RCPSPCPSolver(TimeLimit=15, no_workers=1)



problem_type = "RCPSP"
benchmark_name = "j60.sm"

# SPECIFIC BENCHMARK INSTANCE
# benchmark = load_raw_benchmark(f"raw_data/{problem_type.lower()}/{benchmark_name}")
benchmark = load_benchmark(f"master_thesis_data/{problem_type}/{benchmark_name}")

# cp_solver.solve(benchmark, validate=True, force_execution=True, force_dump=False)
# BRKGA_solver_backward.solve(benchmark, validate=True, force_execution=True, force_dump=False)
# BRKGA_solver_forward.solve(benchmark, validate=True, force_execution=True, force_dump=False)
# naive_GA_solver_backward.solve(benchmark, validate=True, force_execution=True, force_dump=False)
# naive_GA_solver_forward.solve(benchmark, validate=True, force_execution=True, force_dump=False)

# BRKGA_solver_backward_small.solve(benchmark, validate=True, force_execution=True, force_dump=False)
# BRKGA_solver_forward_small.solve(benchmark, validate=True, force_execution=True, force_dump=False)
# naive_GA_solver_backward_small.solve(benchmark, validate=True, force_execution=True, force_dump=False)
# naive_GA_solver_forward_small.solve(benchmark, validate=True, force_execution=True, force_dump=False)

# naive_GA_solver_forward_time_paper.solve(benchmark, validate=True, force_execution=True, force_dump=False)


table_markdown = benchmark.generate_solver_comparison_markdown_table()
table_markdown2 = benchmark.generate_solver_comparison_percent_deviation_markdown_table()

print(table_markdown)
print(table_markdown2)

# benchmark.dump(f"master_thesis_data/{problem_type}/{benchmark_name}")