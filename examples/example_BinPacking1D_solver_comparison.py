## python -m examples.example_BinPacking1D

import numpy as np

from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.core.duplicate import ElementwiseDuplicateElimination

from pymoo.algorithms.soo.nonconvex.brkga import BRKGA

from src.binpacking1d.problem import BinPacking1D
from src.binpacking1d.solvers.solver_cp import BinPacking1DCPSolver
from src.binpacking1d.solvers.solver_ga import BinPacking1DGASolver
from src.general_optimization_solver import load_raw_instance, load_instance, load_raw_benchmark



### naive GA
algorithm = GA(
    pop_size=100,
    n_offsprings=50,
    sampling=FloatRandomSampling(),
    crossover=TwoPointCrossover(prob=0.9),
    mutation=PolynomialMutation(eta=3),
    eliminate_duplicates=True
)
def fitness_func(instance, x, out):
    bins = {}
    for idx, bin_idx in enumerate(x):
        bin_idx = int(bin_idx)
        bins[bin_idx] = bins.get(bin_idx, 0) + instance.weights[idx]
    
    num_bins = len(bins)
    max_bin_load = max(bins.values())
    
    # Objective: Minimize the number of bins used
    out["F"] = num_bins - max_bin_load / instance.bin_capacity
    out["placements"] = x.tolist()
    
    # Constraint: No bin should overflow
    out["G"] = max_bin_load - instance.bin_capacity

    return out

naive_GA_solver = BinPacking1DGASolver(algorithm, fitness_func, ("n_gen", 100), seed=1, solver_name="naive GA")

###

### BRKGA

class MyElementwiseDuplicateElimination(ElementwiseDuplicateElimination):

    def is_equal(self, a, b):
        return (a.X.astype(int) == a.X.astype(int)).all()


# values from https://pymoo.org/algorithms/soo/brkga.html 
algorithm = BRKGA(
    n_elites=30,
    n_offsprings=60,
    n_mutants=10,
    bias=0.7,
    eliminate_duplicates=MyElementwiseDuplicateElimination())

def fitness_func(instance, x, out):
    bins = {}
    for idx, bin_idx in enumerate(x):
        bin_idx = int(bin_idx)
        bins[bin_idx] = bins.get(bin_idx, 0) + instance.weights[idx]
    
    num_bins = len(bins)
    max_bin_load = max(bins.values())
    
    # Objective: Minimize the number of bins used
    out["F"] = num_bins - max_bin_load / instance.bin_capacity
    out["placements"] = x.tolist()
    
    # Constraint: No bin should overflow
    out["G"] = max_bin_load - instance.bin_capacity

    return out
BRKGA_solver = BinPacking1DGASolver(algorithm, fitness_func, ("n_gen", 100), seed=1, solver_name="BRKGA")

###

## CP

cp_solver = BinPacking1DCPSolver(TimeLimit=10)


def indices_to_onehot(indices, num_classes):
    onehot = np.zeros((len(indices), num_classes))
    onehot[np.arange(len(indices)), indices] = 1
    return onehot

#############################################################################


skip_instance_input = True
skip_benchmark_input = False


if not skip_instance_input:
    # SPECIFIC BENCHMARK INSTANCE
    instance = load_raw_instance("raw_data/1d-binpacking/scholl_bin1data/N1C1W1_A.BPP", "")  # , "1Dbinpacking"
    # instance = load_instance("data/1DBINPACKING/scholl_bin1data/N1C1W1_A.json")
    cp_bins_used, cp_solution_variables, cp_solution = cp_solver.solve(instance, validate=False, visualize=False, force_execution=True)
    cp_assignment = cp_solution_variables['item_bin_pos_assignment']


    print("Number of bins used:", cp_bins_used)
    print("Assignment of items to bins:", cp_assignment)
    instance.visualize(cp_assignment)

    ga_fitness_value, ga_assignment, ga_solution = naive_GA_solver.solve(instance)

    if ga_assignment is not None:
        bins = {}
        for idx, bin_idx in enumerate(ga_assignment):
            bin_idx = int(bin_idx)
            bins[bin_idx] = bins.get(bin_idx, 0) + instance.weights[idx]

        num_bins = len(bins)

        ga_assignment = indices_to_onehot(ga_assignment, len(instance.weights))

        print(f"Best solution: {np.floor(ga_solution.X)}")
        print(f"Number of bins used: {num_bins}")
        print(ga_assignment)
        instance.visualize(ga_assignment)

    brkga_fitness_value, brkga_assignment, brkga_solution = BRKGA_solver.solve(instance)


    if brkga_assignment is not None:
        bins = {}
        for idx, bin_idx in enumerate(brkga_assignment):
            bin_idx = int(bin_idx)
            bins[bin_idx] = bins.get(bin_idx, 0) + instance.weights[idx]

        num_bins = len(bins)

        brkga_assignment = indices_to_onehot(brkga_assignment, len(instance.weights))

        print(f"Best solution: {np.floor(brkga_solution.X)}")
        print(f"Number of bins used: {num_bins}")
        print(brkga_assignment)
        instance.visualize(brkga_assignment)


    instance.dump()

if not skip_benchmark_input:
    # SPECIFIC BENCHMARK INSTANCE
    benchmark = load_raw_benchmark("raw_data/1d-binpacking/scholl_bin1data", no_instances=10)
    cp_solver.solve(benchmark, validate=False, visualize=False, force_execution=True)

    naive_GA_solver.solve(benchmark)

    BRKGA_solver.solve(benchmark)
    
    table_markdown = benchmark.generate_solver_comparison_markdown_table()

    print(table_markdown)