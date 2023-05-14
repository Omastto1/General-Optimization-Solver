import docplex.cp.utils_visu as visu
import matplotlib.pyplot as plt
from docplex.cp.model import CpoStepFunction

def visualize(sol, x, no_resources, requests, capacities):
    no_jobs = len(x)
    
    if sol and visu.is_visu_enabled():
        visu.timeline('Solution SchedOptional', 0, 110)
        for job_number in range(no_jobs):
            visu.sequence(name=job_number)
            wt = sol.get_var_solution(x[job_number])
            if wt.is_present():
                if wt.get_start() != wt.get_end():
                    visu.interval(wt, "salmon", x[job_number].get_name())
    visu.show()


    # Define the data for the Gantt chart
    print(sol.get_value(x[0]))
    start_times = [sol.get_var_solution(x[i]).get_start() for i in range(no_jobs)]
    end_times = [sol.get_var_solution(x[i]).get_end() for i in range(no_jobs)]

    # Create the Gantt chart
    fig, ax = plt.subplots()
    for i in range(no_jobs):
        ax.broken_barh([(start_times[i], end_times[i] - start_times[i])], (i, 1), facecolors='blue')
    ax.set_ylim(0, no_jobs)
    ax.set_xlim(0, max(end_times))
    ax.set_xlabel('Time')
    ax.set_yticks(range(no_jobs))
    ax.set_yticklabels(['Activity %d' % i for i in range(no_jobs)])
    ax.grid(True)
    plt.show()

    # https://ibmdecisionoptimization.github.io/docplex-doc/cp/visu.rcpsp.py.html

    if sol and visu.is_visu_enabled():
        load = [CpoStepFunction() for j in range(no_resources)]
        for i in range(no_jobs):
            itv = sol.get_var_solution(x[i])
            for j in range(no_resources):
                if 0 < requests[j][i]:
                    load[j].add_value(itv.get_start(), itv.get_end(), requests[j][i])

        visu.timeline('Solution for RCPSP ') # + filename)
        visu.panel('Tasks')
        for i in range(no_jobs):
            visu.interval(sol.get_var_solution(x[i]), i, x[i].get_name())
        for j in range(no_resources):
            visu.panel('R' + str(j+1))
            visu.function(segments=[(0, 200, capacities[j])], style='area', color='lightgrey')
            visu.function(segments=load[j], style='area', color=j)
        visu.show()