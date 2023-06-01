from src.general_optimization_solver import load_raw_benchmark
from src.solvers.rcpsp import RCPSPSolver

benchmark = load_raw_benchmark("raw_data/rcpsp/j30.sm/", "raw_data/rcpsp/j30opt.sm", "j30", 100)
benchmark.solve(RCPSPSolver(no_workers=14))
