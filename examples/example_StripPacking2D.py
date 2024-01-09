import matplotlib.pyplot as plt
import numpy as np

from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.termination.ftol import SingleObjectiveSpaceTermination
from pymoo.termination.robust import RobustTermination

from src.general_optimization_solver import load_raw_instance, load_instance, load_raw_benchmark

from src.strippacking2d.problem import StripPacking2D
from src.strippacking2d.solvers.solver_cp_not_oriented import StripPacking2DCPSolver
from src.strippacking2d.solvers.solver_cp_oriented import StripPacking2DCPSolver as StripPacking2DCPSolverOriented
from src.strippacking2d.solvers.solver_cp_oriented import StripPacking2DCPSolver as StripPacking2DCPSolverOrientedOptimized
from src.strippacking2d.solvers.ga_solver import StripPacking2DGASolver


skip_custom_input = True
skip_instance_input = True
skip_benchmark_input = True
skip_hybrid = False

# Example usage

if not skip_custom_input:
    rectangles = [{'width': 3, 'height': 4}, {'width':5, 'height': 6}, {'width':2, 'height': 5}]  # Each tuple represents (width, height)
    strip_width = 7

    problem = StripPacking2D(benchmark_name="StripPacking2DTest", instance_name="Test01", data={"rectangles": rectangles, "strip_width": strip_width}, solution={}, run_history=[])


    total_height, placements, solution = StripPacking2DCPSolver(TimeLimit=3).solve(problem, validate=False, visualize=False, force_execution=True)

    print("Total height:", total_height)
    print("Placement of rectangles:", placements)

    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    rectangles = []
    for i, rectangle in enumerate(problem.rectangles):
        x, y = placements[i]
        width, height = rectangle.values()
        rectangles.append((x, y, width, height))

    # Create a figure and axis for plotting
    fig, ax = plt.subplots()
    ax.set_xlim([0, problem.strip_width])
    ax.set_ylim([0, total_height])

    # Draw the large rectangle
    large_rect = Rectangle((0, 0), problem.strip_width, total_height, edgecolor='black', facecolor='none')
    ax.add_patch(large_rect)

    # Draw the small rectangles within the large rectangle
    for x, y, width, height in rectangles:
        print(x, y, width, height)
        rect = Rectangle((x, y), width, height, edgecolor='red', facecolor='green')
        ax.add_patch(rect)

    # Set the aspect ratio and display the plot
    ax.set_aspect('equal', 'box')
    plt.show()


algorithm = GA(
    pop_size=200,
    sampling=FloatRandomSampling(),
    crossover=TwoPointCrossover(prob=1.0),
    mutation=PolynomialMutation(),
    eliminate_duplicates=True
)

def fitness_func(instance, x, out):
    # Calculate the total height based on the order in x
    order = np.argsort(x)
    total_height = 0
    current_width = 0
    current_height = 0
    rectangles = [None] * instance.no_elements
    for i in order:
        if current_width + instance.rectangles[i]['width'] > instance.strip_width:
            total_height += current_height
            current_width = 0
            current_height = 0

        rectangles[i] = (current_width, total_height)
        
        current_width += instance.rectangles[i]['width']
        current_height = max(current_height, instance.rectangles[i]['height'])
    total_height += current_height

    out["F"] = total_height

    # TODO FIX
    out["rectangles"] = rectangles

    return out

if not skip_instance_input:
    # instance = load_raw_instance("raw_data/2d_strip_packing/zdf/zdf2.txt", "")
    instance = load_raw_instance("raw_data/2d_strip_packing/benchmark/BENG01.TXT", "")

    # StripPacking2DGASolver(algorithm, fitness_func, ("n_gen", 10), seed=1).solve(instance)

    
    total_height, placements, solution = StripPacking2DCPSolver(TimeLimit=15).solve(instance, validate=False, visualize=True, force_execution=True)

    
    total_height, placements, solution = StripPacking2DCPSolverOriented(TimeLimit=60).solve(instance, validate=True, visualize=True, force_execution=True)
    total_height, placements, solution = StripPacking2DCPSolverOrientedOptimized(TimeLimit=60).solve(instance, validate=True, visualize=True, force_execution=True)

if not skip_benchmark_input:
    # benchmark = load_raw_benchmark("raw_data/2d_strip_packing/benchmark", "", no_instances=1)
    benchmark = load_raw_benchmark("raw_data/2d_strip_packing/BKW", no_instances=1)
    
    StripPacking2DGASolver(algorithm, fitness_func, ("n_gen", 50), seed=1).solve(benchmark, validate=True, visualize=True, force_execution=True)

    StripPacking2DCPSolver(TimeLimit=15).solve(benchmark, validate=True, visualize=True, force_execution=True)

    table1 = benchmark.generate_solver_comparison_markdown_table()
    table2 = benchmark.generate_solver_comparison_percent_deviation_markdown_table()

    print(table1)
    print(table2)

if not skip_hybrid:
    # instance = load_raw_instance("raw_data/2d_strip_packing/benchmark/BENG01.TXT", "")
    benchmark = load_raw_benchmark("raw_data/2d_strip_packing/BKW", no_instances=1)

    term = RobustTermination(SingleObjectiveSpaceTermination(tol = 0.1), period=30)

    StripPacking2DGASolver(algorithm, fitness_func, term, seed=1).solve(benchmark, validate=True, visualize=True, force_execution=True, hybrid_CP_solver=StripPacking2DCPSolverOriented(TimeLimit=10))
    # StripPacking2DGASolver(algorithm, fitness_func, term, seed=1).solve(benchmark, validate=True, visualize=True, force_execution=True)

    # StripPacking2DCPSolverOriented(TimeLimit=10).solve(benchmark, validate=True, visualize=True, force_execution=True)
