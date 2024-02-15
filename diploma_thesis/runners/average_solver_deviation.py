import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from typing import List

from src.general_optimization_solver import load_benchmark
from src.common.optimization_problem import Benchmark

# Sample data structure
# solvers_data = {
#     'solver1': {
#         'instance1': [(objective_value, time, n_eval), ...],
#         'instance2': [...],
#         ...
#     },
#     'solver2': {
#         ...
#     },
#     ...
# }



def calculate_average_deviations(benchmark: Benchmark, evaluation_points: List[int]):
    """
    computes deviation from lower bound at each step of n_evaluations of GA
    eval_point_max is inclusive maximum
    """
    
    deviations = {}

    # Calculate deviations
    for instance_name, instance in benchmark._instances.items():
        for run in instance._run_history:
            solver = run['solver_name']

            # if not solver.startswith("brkga"):
            #     continue

            # 3 ga vs brkga
            # if not(solver.endswith("0.9_30_5000evals") or solver.startswith("BRKGA 18")):
            #     continue
            
            # ga comp
            # if not (solver.startswith("GA 120_120") and not solver.startswith("GA 120_120_0")):
            #     continue
            
            # brkga comp
            # if not solver.startswith("brkga_TOP_25") and not  solver.startswith("BRKGA"):
            #     continue


            if solver not in deviations:
                deviations[solver] = {n_eval: [] for n_eval in evaluation_points}

            

            lower_bound = instance.critical_path_lower_bound

            obj_values_in_time = []
            eval_point = eval_point_step
            run_history_index = 0
            obj_value = float('inf')
            for eval_point in evaluation_points:

                # print(run['solution_progress'][run_history_index] if run_history_index < len(run['solution_progress']) else run['solution_progress'][-1])
                if run_history_index >= len(run['solution_progress']) or  run['solution_progress'][run_history_index][2] > eval_point:
                    pass
                else:
                    obj_value = run['solution_progress'][run_history_index][0]
                    run_history_index += 1

                deviation = abs(obj_value - lower_bound) / lower_bound * 100
                deviations[solver][eval_point].append(deviation)


    # Average the deviations
    average_deviations = {solver: {} for solver in deviations}
    for solver, n_evals in deviations.items():
        for n_eval, dev_list in n_evals.items():
            average_deviations[solver][n_eval] = np.mean(dev_list)

    return average_deviations



benchmark = load_benchmark(f"master_thesis_data_ga_comp_test/RCPSP/j120.sm")


for instance_name, instance in benchmark._instances.items():
    G = nx.DiGraph()
    for job in range(instance.no_jobs):
        G.add_node(job)
        for predecessor_ in instance.predecessors[job]:
            G.add_edge(job, predecessor_ - 1, weight=-instance.durations[predecessor_ - 1])


    longest_length_paths_negative = nx.single_source_bellman_ford_path_length(
        G, instance.no_jobs - 1)
    instance.critical_path_lower_bound = -longest_length_paths_negative[0]

# Example usage
eval_point_step = 30
eval_point_max = 7000
evaluation_points = list(range(0, eval_point_max+1, eval_point_step))
average_deviations = calculate_average_deviations(benchmark, evaluation_points)


def plot_average_deviations(average_deviations, evaluation_points):
    plt.figure(figsize=(10, 6))

    for solver, deviations in average_deviations.items():
        # Extract the deviations for each evaluation point
        y_values = [deviations[n_eval] for n_eval in evaluation_points]
        plt.plot(evaluation_points, y_values, label=solver)

    plt.xlabel('Number of Evaluations')
    plt.ylabel('Average Deviation')
    plt.title('J120.sm Average Deviations of Solvers Over Evaluations')
    plt.legend()
    plt.grid(True)
    plt.show()

table_markdown2 = benchmark.generate_solver_comparison_percent_deviation_markdown_table(compare_to_cplb=True)

print(table_markdown2)

# {key, for print(average_deviations)
# Assume average_deviations is the output from the previous function
plot_average_deviations(average_deviations, evaluation_points)


'GA 60_60_0.9_30_5000evals': 45.64
'GA 120_120_0.9_30_5000evals': 43.96
'GA 240_240_0.9_30_5000evals': 43.84
'BRKGA 24_78_18_0.7_5000evals': 42.48
'BRKGA 12_72_36_0.7_5000evals': 43.41
'BRKGA 18_76_26_0.7_5000evals': 43.0
'brkga_TOP_25%_BOT_20% 30_66_24_0.7_5000evals': 43.13
'brkga_TOP_25%_BOT_15% 30_72_18_0.7_5000evals': 42.5
'GA 120_120_SinglePointCrossover_30_5000evals': 44.9
'GA 120_120_TwoPointCrossover_30_5000evals': 43.96
'GA 120_120_SBX_30_5000evals': 42.82
'GA 120_120_UniformCrossover_30_5000evals': 42.46
'GA 120_120_HalfUniformCrossover_30_5000evals': 42.42
'brkga_TOP_10%_BOT_20% 360_2520_720_0.7_5000evals': 48.71
