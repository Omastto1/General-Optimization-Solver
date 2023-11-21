import numpy as np

from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.core.duplicate import ElementwiseDuplicateElimination

from pymoo.algorithms.soo.nonconvex.brkga import BRKGA

from src.rcpsp.problem import RCPSP
from src.rcpsp.solvers.solver_cp import RCPSPCPSolver
from src.rcpsp.solvers.solver_ga import RCPSPGASolver
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
    # project makespan unknown, default to 65536
    finish_times = np.full(instance.no_jobs, 65536)
    
    # Start with the ending node
    finish_times[-1] = 0
    
    # List to track unscheduled jobs. Initially, it contains all jobs except the ending node.
    unscheduled_jobs = list(range(instance.no_jobs - 1))
    resource_usage_over_time = np.zeros((instance.no_renewable_resources, sum(instance.durations)))
    
    # While there are unscheduled jobs
    while len(unscheduled_jobs) > 0:
        # Find jobs that can be scheduled (those whose all successors are already scheduled)
        schedulable_jobs = [j for j in unscheduled_jobs if all((succ - 1) not in unscheduled_jobs for succ in instance.successors[j])]

        # Sort schedulable jobs based on their order in X (in reverse since we're scheduling backward)
        schedulable_jobs.sort(key=lambda j: -x[j])

        for job in schedulable_jobs:
            # Find the latest time this job can be finished based on resource availability and successor constraints
            finish_time = min([finish_times[succ-1] - instance.durations[succ-1] for succ in instance.successors[job]])
            while True:
                # Check if finishing the job at 'finish_time' violates any resource constraints
                resource_violation = False
                for t in range(finish_time - instance.durations[job], finish_time):
                    for k in range(instance.no_renewable_resources):
                        if resource_usage_over_time[k, t] + instance.requests[k][job] > instance.renewable_capacities[k]:
                            resource_violation = True
                            break
                    if resource_violation:
                        break
                
                # If there's no violation, break. Otherwise, try the previous time unit.
                if not resource_violation:
                    break
                finish_time -= 1
            
            # Schedule the job and update resource usage
            finish_times[job] = finish_time
            
            for t in range(finish_time - instance.durations[job], finish_time):
                for k in range(instance.no_renewable_resources):
                    resource_usage_over_time[k, t] += instance.requests[k][job]
            
            # Remove the scheduled job from the list of unscheduled jobs
            unscheduled_jobs.remove(job)

    # Shift all times so that the starting node starts at 0
    shift = finish_times[0] - instance.durations[0]
    finish_times -= shift
    start_times = finish_times - instance.durations

    # Calculate makespan and constraints violation
    makespan = np.max(finish_times)
    resource_violations = np.max(resource_usage_over_time - np.array(instance.renewable_capacities)[:, np.newaxis], axis=1)

    if len(resource_violations) > 1:
        pass
        # print("asd")
    
    out["F"] = makespan
    out["G"] = resource_violations
    out["start_times"] = start_times

    return out

naive_GA_solver = RCPSPGASolver(algorithm, fitness_func, ("n_gen", 100), seed=1, solver_name="naive GA")

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


BRKGA_solver = RCPSPGASolver(algorithm, fitness_func, ("n_gen", 100), seed=1, solver_name="BRKGA")

###

## CP

cp_solver = RCPSPCPSolver(TimeLimit=10)


def indices_to_onehot(indices, num_classes):
    onehot = np.zeros((len(indices), num_classes))
    onehot[np.arange(len(indices)), indices] = 1
    return onehot

#############################################################################


skip_instance_input = True
skip_benchmark_input = False


if not skip_instance_input:
    # SPECIFIC BENCHMARK INSTANCE
    instance = load_raw_instance("raw_data/rcpsp/CV/cv1.rcp", "")  # , "1Dbinpacking"
    # instance = load_instance("data/1DBINPACKING/scholl_bin1data/N1C1W1_A.json")
    cp_bins_used, cp_solution_variables, cp_solution = cp_solver.solve(instance, validate=True, visualize=True, force_execution=True)
    cp_assignment = cp_solution_variables['tasks_schedule']


    print("Number of bins used:", cp_bins_used)
    print("tasks_schedule:", cp_assignment)

    ga_fitness_value, ga_assignment, ga_solution = naive_GA_solver.solve(instance, visualize=True, validate=True)

    brkga_fitness_value, brkga_assignment, brkga_solution = BRKGA_solver.solve(instance, visualize=True, validate=True)


    instance.dump()

if not skip_benchmark_input:
    # SPECIFIC BENCHMARK INSTANCE
    benchmark = load_raw_benchmark("raw_data/rcpsp/CV", no_instances=10)
    cp_solver.solve(benchmark, validate=False, visualize=False, force_execution=True)

    naive_GA_solver.solve(benchmark)

    BRKGA_solver.solve(benchmark)
    
    table_markdown = benchmark.generate_solver_comparison_markdown_table()

    print(table_markdown)