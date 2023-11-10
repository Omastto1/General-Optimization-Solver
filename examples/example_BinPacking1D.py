## python -m examples.example_BinPacking1D

import numpy as np

from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.pm import PolynomialMutation

from src.binpacking1d.problem import BinPacking1D
from src.binpacking1d.solvers.solver_cp import BinPacking1DCPSolver
from src.binpacking1d.solvers.solver_ga import BinPacking1DGASolver
from src.general_optimization_solver import load_raw_instance, load_instance, load_raw_benchmark

# GA ALG CONFIG AND FITNESS FUNC
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

def indices_to_onehot(indices, num_classes):
    onehot = np.zeros((len(indices), num_classes))
    onehot[np.arange(len(indices)), indices] = 1
    return onehot


skip_custom_input = False
skip_instance_input = False
skip_benchmark_input = False

if not skip_custom_input:
    #### CUSTOM INPUT

    weights = [2, 5, 4, 7, 1, 3, 8]
    bin_capacity = 10
    problem = BinPacking1D(benchmark_name="BinPacking1DTest", instance_name="Test01", data={"weights": weights, "bin_capacity": bin_capacity}, solution={}, run_history=[])

    cp_bins_used, cp_assignment, cp_solution = BinPacking1DCPSolver().solve(problem, validate=False, visualize=True, force_execution=True)

    print("Number of bins used:", cp_bins_used)
    print("Assignment of items to bins:", cp_assignment)

    ga_fitness_value, ga_assignment, ga_solution = BinPacking1DGASolver(algorithm, fitness_func, ("n_gen", 100), seed=1).solve(problem)

    bins = {}
    for idx, bin_idx in enumerate(ga_assignment):
        bin_idx = int(bin_idx)
        bins[bin_idx] = bins.get(bin_idx, 0) + problem.weights[idx]

    num_bins = len(bins)


    ga_assignment = indices_to_onehot(ga_assignment, len(weights))

    print(f"Best solution: {np.floor(ga_solution.X)}")
    print(f"Number of bins used: {num_bins}")
    problem.visualize(ga_assignment)


if not skip_instance_input:
    # SPECIFIC BENCHMARK INSTANCE
    instance = load_raw_instance("raw_data/1d-binpacking/scholl_bin1data/N1C1W1_A.BPP", "", "1Dbinpacking")
    # instance = load_instance("data/1DBINPACKING/scholl_bin1data/N1C1W1_A.json")
    cp_bins_used, cp_assignment, cp_solution = BinPacking1DCPSolver(TimeLimit=10).solve(instance, validate=False, visualize=False, force_execution=True)


    print("Number of bins used:", cp_bins_used)
    print("Assignment of items to bins:", cp_assignment)
    instance.visualize(cp_assignment)

    ga_fitness_value, ga_assignment, ga_solution = BinPacking1DGASolver(algorithm, fitness_func, ("n_gen", 100), seed=1).solve(instance)


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

    instance.dump()

if not skip_benchmark_input:
    # SPECIFIC BENCHMARK INSTANCE
    benchmark = load_raw_benchmark("raw_data/1d-binpacking/scholl_bin1data", "", "1Dbinpacking", 2)
    # benchmark = load_benchmark("data/1DBINPACKING/scholl_bin1data/N1C1W1_A.json")
    BinPacking1DCPSolver(TimeLimit=2).solve(benchmark, validate=False, visualize=False, force_execution=True)


    # print("Number of bins used:", cp_bins_used)
    # print("Assignment of items to bins:", cp_assignment)
    # benchmark.visualize(cp_assignment)

    BinPacking1DGASolver(algorithm, fitness_func, ("n_gen", 20), seed=1).solve(benchmark)


    # bins = {}
    # for idx, bin_idx in enumerate(ga_assignment):
    #     bin_idx = int(bin_idx)
    #     bins[bin_idx] = bins.get(bin_idx, 0) + instance.weights[idx]

    # num_bins = len(bins)

    # ga_assignment = indices_to_onehot(ga_assignment, len(instance.weights))

    # print(f"Best solution: {np.floor(ga_solution.X)}")
    # print(f"Number of bins used: {num_bins}")
    # print(ga_assignment)
    # instance.visualize(ga_assignment)

    # instance.dump()
    
    table_markdown = benchmark.generate_solver_comparison_markdown_table()