import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 

import matplotlib.pyplot as plt
import numpy as np

from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.pm import PolynomialMutation

from src.general_optimization_solver import load_raw_instance, load_instance, load_raw_benchmark, load_benchmark

from src.strippacking2d.problem import StripPacking2D
from src.strippacking2d.solvers.solver_cp_not_oriented import StripPacking2DCPSolver
from src.strippacking2d.solvers.solver_cp_oriented import StripPacking2DCPSolver as StripPacking2DCPSolverOriented
from src.strippacking2d.solvers.ga_solver import StripPacking2DGASolver
from pymoo.termination.robust import RobustTermination
from pymoo.termination.ftol import SingleObjectiveSpaceTermination


from ga_fitness_functions.strip_packing_2d.leveled import fitness_func


term = RobustTermination(SingleObjectiveSpaceTermination(tol = 0.1), period=20)
algorithm = GA(
    pop_size=200,
    sampling=FloatRandomSampling(),
    crossover=TwoPointCrossover(prob=1.0),
    mutation=PolynomialMutation(),
    eliminate_duplicates=True
)

problem_type = "2d_strip_packing"
benchmark_name = "BKW"

# benchmark = load_raw_benchmark("raw_data/2d_strip_packing/benchmark", "", no_instances=1)
# benchmark = load_raw_benchmark(f"raw_data/{problem_type}/{benchmark_name}")
benchmark = load_benchmark(f"master_thesis_data/{problem_type}/{benchmark_name}")
# benchmark = load_instance(f"master_thesis_data/{problem_type}/{benchmark_name}/3.json")


# StripPacking2DGASolver(algorithm, fitness_func, term, seed=1, solver_name="naive GA 200_1.0").solve(benchmark, validate=True, visualize=False, force_execution=True, force_dump=False)

# StripPacking2DCPSolver(TimeLimit=15, no_workers=1).solve(benchmark, validate=True, visualize=False, force_execution=True, force_dump=False)
# StripPacking2DCPSolver(TimeLimit=60, no_workers=1).solve(benchmark, validate=True, visualize=False, force_execution=True, force_dump=False)

# StripPacking2DGASolver(algorithm, fitness_func, ("n_gen", 20), seed=1).solve(benchmark, validate=True, visualize=False, force_execution=True, hybrid_CP_solver=StripPacking2DCPSolver(TimeLimit=15, no_workers=1), force_dump=False)

# StripPacking2DGASolver(algorithm, fitness_func, ("n_gen", 20), seed=1).solve(benchmark, validate=True, visualize=False, force_execution=True, hybrid_CP_solver=StripPacking2DCPSolver(TimeLimit=60, no_workers=1), force_dump=False)

# StripPacking2DGASolver(algorithm, fitness_func, term, seed=1).solve(benchmark, validate=True, visualize=False, force_execution=True, hybrid_CP_solver=StripPacking2DCPSolverOriented(TimeLimit=10, no_workers=1), force_dump=False)

# StripPacking2DGASolver(algorithm, fitness_func, ("n_gen", 20), seed=1).solve(benchmark, validate=True, visualize=False, force_execution=True, hybrid_CP_solver=StripPacking2DCPSolverOriented(TimeLimit=10), force_dump=False)

table1 = benchmark.generate_solver_comparison_markdown_table()
table2 = benchmark.generate_solver_comparison_percent_deviation_markdown_table()

print(table1)
print(table2)

# benchmark.dump(f"master_thesis_data/{problem_type}/{benchmark_name}")