import numpy as np


def fitness_func_forward(instance, x, out):
    # project makespan unknown, default to 65536
    start_times = np.full(instance.no_jobs, 0)
    selected_modes = {0: 0}
    max_end_time = sum(max(modes_durations) for modes_durations in instance.durations)
    
    # Start with the ending node
    start_times[0] = 0
    
    # List to track unscheduled jobs. Initially, it contains all jobs except the ending node.
    unscheduled_jobs = list(range(1, instance.no_jobs))
    renewable_resource_usage_over_time = np.zeros((instance.no_renewable_resources, max_end_time))
    non_renewable_resource_usage_over_time = np.zeros((instance.no_non_renewable_resources, max_end_time))
    
    infeasible = False

    # While there are unscheduled jobs
    while len(unscheduled_jobs) > 0 and not infeasible:
        # Find jobs that can be scheduled (those whose all predecessors are already scheduled)
        schedulable_jobs = [j for j in unscheduled_jobs if all((pred - 1) not in unscheduled_jobs for pred in instance.predecessors[j])]

        schedulable_jobs.sort(key=lambda j: x[j*4])  # 1st job priority, 1st job 3 mode priorities, 2st job, 2 mode priorities of 2nd job, ...

        for job in schedulable_jobs:
            # Find the latest time this job can be finished based on resource availability and predecessor constraints
            _start_times = []
            for pred in instance.predecessors[job]:
                if pred - 1 == 0:
                    pred_selected_mode = 0
                else:
                    pred_selected_mode = x[(pred - 1) * 4 + 1: (pred - 1) * 4 + 4].argmax()
                _start_times.append(start_times[pred-1] + instance.durations[pred-1][pred_selected_mode])

            if job == instance.no_jobs - 1:
                job_selected_mode = 0
            else:
                job_selected_mode = x[job*4 + 1: job*4 + 4].argmax()
            selected_modes[job] = job_selected_mode
            start_time = max(_start_times)
            while True:
                # Check if finishing the job at 'start_time' violates any resource constraints
                resource_violation = False
                for t in range(start_time, start_time + instance.durations[job][job_selected_mode]):
                    if start_time + instance.durations[job][job_selected_mode] > max_end_time:
                        resource_violation = True
                        break

                    for k in range(instance.no_renewable_resources):
                        if renewable_resource_usage_over_time[k, t] + instance.requests[k][job][job_selected_mode] > instance.renewable_capacities[k]:
                            resource_violation = True
                            break
                    if resource_violation:
                        break
                
                t = start_time
                while not resource_violation and t < max_end_time:
                    for k in range(instance.no_non_renewable_resources):
                        if non_renewable_resource_usage_over_time[k, t] + instance.requests[k + instance.no_renewable_resources][job][job_selected_mode] > instance.non_renewable_capacities[k]:
                            resource_violation = True
                            break
                    t += 1
                
                # If there's no violation, break. Otherwise, try the following time unit.
                if not resource_violation:
                    break
                start_time += 1

                if start_time > max_end_time:
                    infeasible = True
                    break
            
            if infeasible:
                break

            # Schedule the job and update resource usage
            start_times[job] = start_time
            
            for t in range(start_time, start_time + instance.durations[job][job_selected_mode]):
                for k in range(instance.no_renewable_resources):
                    renewable_resource_usage_over_time[k, t] += instance.requests[k][job][job_selected_mode]
            
            for t in range(start_time, max_end_time):
                for k in range(instance.no_non_renewable_resources):
                    non_renewable_resource_usage_over_time[k, t] += instance.requests[k + instance.no_renewable_resources][job][job_selected_mode]


            # Remove the scheduled job from the list of unscheduled jobs
            unscheduled_jobs.remove(job)

    if not infeasible:
        # Calculate makespan and constraints violation
        makespan = np.max([start_times[job] + duration[selected_modes[job]] for job, duration in enumerate(instance.durations)])
        resource_violations = np.max(renewable_resource_usage_over_time - np.array(instance.renewable_capacities)[:, np.newaxis], axis=1).tolist()
        resource_violations += np.max(non_renewable_resource_usage_over_time - np.array(instance.non_renewable_capacities)[:, np.newaxis], axis=1).tolist()
    else:
        makespan = -1
        resource_violations = [1] * (instance.no_renewable_resources + instance.no_non_renewable_resources)

    
    out["F"] = makespan
    out["G"] = resource_violations
    out["start_times"] = start_times
    out["selected_modes"] = selected_modes

    return out
