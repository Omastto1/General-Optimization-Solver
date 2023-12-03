from src.general_optimization_solver import load_raw_benchmark
from src.jobshop.solver import JobShopSolver


benchmark = load_raw_benchmark("raw_data/jobshop/jobshop", "raw_data/jobshop/instances_results.txt", "jobshop")
benchmark.solve(JobShopSolver(no_workers=14))
