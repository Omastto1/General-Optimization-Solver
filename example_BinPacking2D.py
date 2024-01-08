from src.binpacking2d.problem import BinPacking2D
from src.binpacking2d.solvers.solver_cp import BinPacking2DCPSolver
from src.general_optimization_solver import load_raw_instance


# Example usage
# rectangles = [(3, 2), (5, 4), (2, 6), (4, 4), (5, 5), (4, 3)]
# bin_size = (10, 10)
# problem = BinPacking2D(benchmark_name="BinPacking2DTest", instance_name="Test01", data={"items_sizes": rectangles, "bin_size": bin_size}, solution={}, run_history={})


# bins_used, assignment, orientations = BinPacking2DCPSolver().solve(problem, validate=False, visualize=False, force_execution=True)

# problem.visualize(assignment, orientations)

# print("Number of bins used:", bins_used)
# print("Assignment of rectangles to bins:", assignment)
# print("Orientations of rectangles:", orientations)



# SPECIFIC BENCHMARK INSTANCE
instance = load_raw_instance("data/2DBINPACKING/ngcut_bin/ngcut_1.BPP", "", "2Dbinpacking")
print(instance.__dict__)
bins_used, assignment, orientations = BinPacking2DCPSolver().solve(instance, validate=False, visualize=False, force_execution=True)

instance.visualize(assignment, orientations)

print("Number of bins used:", bins_used)
print("Assignment of rectangles to bins:", assignment)
print("Orientations of rectangles:", orientations)
