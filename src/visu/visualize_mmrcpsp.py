import docplex.cp.utils_visu as visu
import matplotlib.pyplot as plt
from docplex.cp.model import CpoStepFunction

def visualize(sol, x, no_renewable_resources, no_non_renewable_resources, requests, renewable_capacities, non_renewable_capacities):
    no_jobs = len(x)
    # https://ibmdecisionoptimization.github.io/docplex-doc/cp/visu.rcpsp.py.html

    
    if sol and visu.is_visu_enabled():
        load = [CpoStepFunction() for j in range(no_renewable_resources + no_non_renewable_resources)]
        for i in range(no_jobs):
            itv = sol.get_var_solution(x[i])
            mode = int(x[i].get_name().split("_")[-1])
            for j in range(no_renewable_resources):
                if 0 < requests[j][i][mode]:
                    load[j].add_value(itv.get_start(), itv.get_end(), requests[j][i][mode])
            for j in range(2, no_non_renewable_resources + 2):
                if 0 < requests[j][i][mode]:
                    load[j].add_value(itv.get_start(), sol.get_objective_value(), requests[j][i][mode])

        visu.timeline('Solution for RCPSP ') # + filename)
        visu.panel('Tasks')
        for i in range(no_jobs):
            visu.interval(sol.get_var_solution(x[i]), i, x[i].get_name())

        for j in range(no_renewable_resources):
            visu.panel('R' + str(j+1))
            visu.function(segments=[(0, 200, renewable_capacities[j])], style='area', color='lightgrey')
            visu.function(segments=load[j], style='area', color=j)

        for j in range(2, no_non_renewable_resources + 2):
            visu.panel('NR' + str(j+1))
            visu.function(segments=[(0, 200, non_renewable_capacities[j-2])], style='area', color='lightgrey')
            visu.function(segments=load[j], style='area', color=j)

        visu.show()