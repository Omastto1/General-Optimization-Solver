from src.general_optimization_solver import load_raw_benchmark
from src.solvers.rcpsp import RCPSPSolver

benchmark = load_raw_benchmark("raw_data/rcpsp/DC2-npv75/", "raw_data/rcpsp/DC2-npv75.xlsx", "patterson", 100)
benchmark.solve(RCPSPSolver(no_workers=14))
