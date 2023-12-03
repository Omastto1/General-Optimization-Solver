import numpy as np

from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.pm import PolynomialMutation

from src.rcpsp.solvers.solver_cp import RCPSPCPSolver
from src.rcpsp.solvers.solver_ga import RCPSPGASolver
from src.general_optimization_solver import load_raw_instance, load_instance, load_raw_benchmark



### naive GA
algorithm = GA(
    pop_size=120,
    n_offsprings=120,
    sampling=FloatRandomSampling(),
    crossover=TwoPointCrossover(prob=0.9),
    mutation=PolynomialMutation(eta=20),
    eliminate_duplicates=True
)

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

naive_GA_solver_forward = RCPSPGASolver(algorithm, fitness_func_forward, ("n_gen", 5), seed=1, solver_name="naive GA forward")

###

## CP

cp_solver = RCPSPCPSolver(TimeLimit=15)

#############################################################################


# SPECIFIC BENCHMARK INSTANCE
instance = load_raw_instance("raw_data/rcpsp/CV/cv1.rcp", "")

brkga_fitness_value, brkga_assignment, brkga_solution = naive_GA_solver_forward.solve(instance, visualize=False, validate=True)

cp_bins_used, cp_solution_variables, cp_solution = cp_solver._solve(instance, validate=True, visualize=True, force_execution=True, initial_solution=brkga_assignment)

instance.dump()
