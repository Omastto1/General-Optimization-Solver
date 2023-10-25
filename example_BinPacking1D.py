from src.binpacking1d.problem import BinPacking1D
from src.binpacking1d.solver import BinPacking1DSolver


# Example usage
weights = [2, 5, 4, 7, 1, 3, 8]
bin_capacity = 10
problem = BinPacking1D(benchmark_name="BinPacking1DTest", instance_name="Test01", data={"weights": weights, "bin_capacity": bin_capacity}, solution={}, run_history={})


bins_used, assignment = BinPacking1DSolver()._solve_cp(problem, validate=False, visualize=False, force_execution=True)

problem.visualize(assignment)

print("Number of bins used:", bins_used)
print("Assignment of items to bins:", assignment)
