from .src.general_optimization_solver import load_raw_benchmark
from src.solvers.rcpsp import RCPSPSolver

benchmark = load_raw_benchmark("raw_data/rcpsp/cv/", "raw_data/rcpsp/CV.xlsx", "patterson", 100)
benchmark.solve(RCPSPSolver(no_workers=14))
