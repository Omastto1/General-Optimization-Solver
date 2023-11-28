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
from src.general_optimization_solver import load_raw_instance, load_instance, load_raw_benchmark, load_benchmark



### naive GA
algorithm = GA(
    pop_size=120,
    n_offsprings=120,
    sampling=FloatRandomSampling(),
    crossover=TwoPointCrossover(prob=0.9),
    mutation=PolynomialMutation(eta=20),
    eliminate_duplicates=True
)


def fitness_func_backward(instance, x, out):
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
            finish_time = min(finish_times[succ-1] - instance.durations[succ-1] for succ in instance.successors[job])
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

# naive_GA_solver = RCPSPGASolver(algorithm, fitness_func_backward, ("n_gen", 100), seed=1, solver_name="naive GA")

def fitness_func_forward(instance, x, out):
    # project makespan unknown, default to 65536
    start_times = np.full(instance.no_jobs, 0)
    
    # Start with the ending node
    start_times[-1] = 0
    
    # List to track unscheduled jobs. Initially, it contains all jobs except the ending node.
    unscheduled_jobs = list(range(1, instance.no_jobs))
    resource_usage_over_time = np.zeros((instance.no_renewable_resources, sum(instance.durations)))
    
    # While there are unscheduled jobs
    while len(unscheduled_jobs) > 0:
        # Find jobs that can be scheduled (those whose all predecessors are already scheduled)
        schedulable_jobs = [j for j in unscheduled_jobs if all((pred - 1) not in unscheduled_jobs for pred in instance.predecessors[j])]

        # Sort schedulable jobs based on their order in X -------- (in reverse since we're scheduling backward)
        schedulable_jobs.sort(key=lambda j: x[j])

        for job in schedulable_jobs:
            # Find the latest time this job can be finished based on resource availability and predecessor constraints
            start_time = max(start_times[pred-1] + instance.durations[pred-1] for pred in instance.predecessors[job])
            while True:
                # Check if finishing the job at 'start_time' violates any resource constraints
                resource_violation = False
                for t in range(start_time, start_time + instance.durations[job]):
                    for k in range(instance.no_renewable_resources):
                        if resource_usage_over_time[k, t] + instance.requests[k][job] > instance.renewable_capacities[k]:
                            resource_violation = True
                            break
                    if resource_violation:
                        break
                
                # If there's no violation, break. Otherwise, try the following time unit.
                if not resource_violation:
                    break
                start_time += 1
            
            # Schedule the job and update resource usage
            start_times[job] = start_time
            
            for t in range(start_time, start_time + instance.durations[job]):
                for k in range(instance.no_renewable_resources):
                    resource_usage_over_time[k, t] += instance.requests[k][job]
            
            # Remove the scheduled job from the list of unscheduled jobs
            unscheduled_jobs.remove(job)

    # Shift all times so that the starting node starts at 0
    # shift = start_times[0] - instance.durations[0]
    # start_times -= shift
    # start_times = start_times - instance.durations

    # Calculate makespan and constraints violation
    makespan = np.max(start_times + instance.durations)
    resource_violations = np.max(resource_usage_over_time - np.array(instance.renewable_capacities)[:, np.newaxis], axis=1)

    if len(resource_violations) > 1:
        pass
        # print("asd")
    
    out["F"] = makespan
    out["G"] = resource_violations
    out["start_times"] = start_times

    return out


naive_GA_solver_backward = RCPSPGASolver(algorithm, fitness_func_backward, ("n_gen", 5), seed=1, solver_name="naive GA backward")
naive_GA_solver_forward = RCPSPGASolver(algorithm, fitness_func_forward, ("n_gen", 5), seed=1, solver_name="naive GA forward")

###

### BRKGA

class MyElementwiseDuplicateElimination(ElementwiseDuplicateElimination):

    def is_equal(self, a, b):
        return (a.X.astype(int) == b.X.astype(int)).all()


# values from https://pymoo.org/algorithms/soo/brkga.html 
algorithm = BRKGA(
    n_elites=40,
    n_offsprings=78,
    n_mutants=30,
    bias=0.7,
    eliminate_duplicates=MyElementwiseDuplicateElimination())


BRKGA_solver_forward = RCPSPGASolver(algorithm, fitness_func_forward, ("n_gen", 2), seed=1, solver_name="BRKGA_backward3")
BRKGA_solver_backward = RCPSPGASolver(algorithm, fitness_func_backward, ("n_gen", 2), seed=1, solver_name="BRKGA_forward3")

###

## CP

cp_solver = RCPSPCPSolver(TimeLimit=15)


def indices_to_onehot(indices, num_classes):
    onehot = np.zeros((len(indices), num_classes))
    onehot[np.arange(len(indices)), indices] = 1
    return onehot

#############################################################################


skip_instance_input = True
skip_benchmark_input = False

benchmark = load_benchmark("data/RCPSP/RG30_Set 1", no_instances=5)  # , "1Dbinpacking"


table_markdown = benchmark.generate_solver_comparison_markdown_table()
table_markdown2 = benchmark.generate_solver_comparison_percent_deviation_markdown_table()

print(table_markdown)
print(table_markdown2)




if not skip_instance_input:
    # SPECIFIC BENCHMARK INSTANCE
    instance = load_raw_instance("raw_data/rcpsp/CV/cv1.rcp", "")  # , "1Dbinpacking"
    # instance = load_instance("data/1DBINPACKING/scholl_bin1data/N1C1W1_A.json")

    cp_bins_used, cp_solution_variables, cp_solution = cp_solver.solve(instance, validate=True, visualize=True, force_execution=True)
    # cp_assignment = cp_solution_variables['tasks_schedule']


    # print("Number of bins used:", cp_bins_used)
    # print("tasks_schedule:", cp_assignment)

    # ga_fitness_value, ga_assignment, ga_solution = naive_GA_solver.solve(instance, visualize=True, validate=True)

    brkga_fitness_value, brkga_assignment, brkga_solution = BRKGA_solver.solve(instance, visualize=False, validate=True)


    instance.dump()

if not skip_benchmark_input:
    # SPECIFIC BENCHMARK INSTANCE
    # benchmark = load_raw_benchmark("raw_data/rcpsp/CV", no_instances=10)
    # benchmark = load_raw_benchmark("raw_data/rcpsp/j120.sm", no_instances=10)  # , "1Dbinpacking"
    # benchmark = load_raw_benchmark("raw_data/rcpsp/RG30_Set 1", no_instances=5)  # , "1Dbinpacking"
    # instance_ = load_raw_instance("raw_data/rcpsp/j120.sm/j1201_1.sm", "")

    # cp_solver.solve(benchmark, validate=True, force_execution=True)

    # BRKGA_solver_backward.solve(benchmark, validate=True, force_execution=True)
    BRKGA_solver_forward.solve(benchmark, validate=True, force_execution=True)

    # naive_GA_solver_backward.solve(benchmark, validate=True, force_execution=True)
    naive_GA_solver_forward.solve(benchmark, validate=True, force_execution=True)

    
    table_markdown = benchmark.generate_solver_comparison_markdown_table()
    table_markdown2 = benchmark.generate_solver_comparison_percent_deviation_markdown_table()

    print(table_markdown)
    print(table_markdown2)