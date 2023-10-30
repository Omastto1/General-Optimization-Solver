import numpy as np

from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.pm import PolynomialMutation

from src.general_optimization_solver import load_raw_instance, load_instance
from src.rcpsp.solvers.solver_ga import RCPSPGASolver
from src.rcpsp.solvers.solver_cp import RCPSPCPSolver


## python -m examples.example_rcpsp


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
        schedulable_jobs = [j for j in unscheduled_jobs if all(succ not in unscheduled_jobs for succ in instance.successors[j])]

        # Sort schedulable jobs based on their order in X (in reverse since we're scheduling backward)
        schedulable_jobs.sort(key=lambda j: -x[j])

        for job in schedulable_jobs:
            # Find the latest time this job can be finished based on resource availability and successor constraints
            finish_time = min([finish_times[succ - 1] - instance.durations[succ-1] for succ in instance.successors[job]])
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
    
    out["F"] = makespan
    out["G"] = resource_violations
    out["start_times"] = start_times


# instance = load_raw_instance("raw_data/rcpsp/CV/cv1.rcp", "raw_data/rcpsp/CV.xlsx", "patterson")
instance = load_instance("data/RCPSP/CV/cv1.json")

# Define the algorithm
algorithm = GA(
    pop_size=100,
    n_offsprings=50,
    sampling=FloatRandomSampling(),
    crossover=TwoPointCrossover(prob=0.9),
    mutation=PolynomialMutation(eta=3),
    eliminate_duplicates=True
)

# ga_fitness_value, ga_startimes, ga_solution = RCPSPGASolver().solve(algorithm, instance, fitness_func, ("n_gen", 100), validate=True, visualize=True)
# print("Best solution found: \nX = ", ga_solution.X)

cp_solution, cp_variables = RCPSPCPSolver(TimeLimit=10).solve(instance, validate=True, visualize=True)


ga_fitness_value, ga_startimes, ga_solution = RCPSPGASolver().solve(algorithm, instance, fitness_func, ("n_gen", 100), validate=True, visualize=True)

instance.dump()
