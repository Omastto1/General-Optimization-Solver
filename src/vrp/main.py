import numpy
from docplex.mp.model import Model
from docplex.cp.model import CpoModel
import numpy as np
import csv
import cplex

from matplotlib import pyplot as plt

from loader import *
from data_model import VRPTWInstance


def main():
    path = 'VRP\\solomon_25\\C101.txt'
    # data = load_instance(path, 100)
    data = random_instance(16, 100)
    print(data)
    # solve_vrp1(data)
    # VRP_ILP_formulation(data)
    VRPTW_CP_formulation(data)


def VRP_ILP_formulation(data: VRPTWInstance):
    n = len(data.demands) - 1
    Q = data.vehicle_capacities
    N = [i for i in range(1, n + 1)]
    V = [0] + N
    d = data.demands

    plt.scatter(data.loc_x[1:], data.loc_y[1:], c='b')
    plt.plot(data.loc_x[0], data.loc_y[0], c='r', marker='s')
    for i in N:
        plt.annotate('$d_%d=%d$' % (i, d[i]), (data.loc_x[i] + 2, data.loc_y[i]))
    plt.show()

    mdl = Model('CVRP')
    u = mdl.continuous_var_dict(N, ub=Q, name='u')

    A = [(i, j) for i in V for j in V if i != j]
    x = mdl.binary_var_dict(A, name='x')
    c = {(i, j): np.hypot(data.loc_x[i] - data.loc_x[j], data.loc_y[i] - data.loc_y[j]) for i, j in A}
    mdl.minimize(mdl.sum(c[i, j] * x[i, j] for i, j in A))
    mdl.add_constraints(mdl.sum(x[i, j] for j in V if j != i) == 1 for i in N)
    mdl.add_constraints(mdl.sum(x[i, j] for i in V if i != j) == 1 for j in N)
    mdl.add_indicator_constraints_(
        mdl.indicator_constraint(x[i, j], u[i] + d[j] == u[j]) for i, j in A if i != 0 and j != 0)
    mdl.add_constraints(u[i] >= d[i] for i in N)

    mdl.parameters.timelimit = 15  # Add running time limit

    solution = mdl.solve(log_output=True)
    print(solution)
    active_arcs = [a for a in A if x[a].solution_value > 0.9]
    plt.scatter(data.loc_x[1:], data.loc_y[1:], c='b')
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    c = 0
    for i in N:
        plt.annotate('$d_%d=%d$' % (i, d[i]), (data.loc_x[i] + 2, data.loc_y[i]))
    for i, j in active_arcs:
        plt.plot([data.loc_x[i], data.loc_x[j]], [data.loc_y[i], data.loc_y[j]], c=colors[c], alpha=0.3)
    c += 1
    if c >= len(colors):
        c = 0
    plt.plot(data.loc_x[0], data.loc_y[0], c='r', marker='s')
    plt.show()


def VRPTW_CP_formulation(data: VRPTWInstance):
    mdl = CpoModel()

    firstDepotVisit = 0
    # copy the depot to end of list
    numpy.append(data.time_windows, data.time_windows[0])
    numpy.append(data.service_times, data.service_times[0])
    numpy.append(data.demands, data.demands[0])
    numpy.append(data.loc_x, data.loc_x[0])
    numpy.append(data.loc_y, data.loc_y[0])

    n = len(data.demands) - 1
    q = len(data.vehicle_capacities) - 1

    dist = {(i, j): np.hypot(data.loc_x[i] - data.loc_x[j], data.loc_y[i] - data.loc_y[j]) for i in
            range(len(data.loc_x)) for j in range(len(data.loc_x))}

    visitInterval = {visit: mdl.interval_var(size=n, name=f'visitInterval_{visit}')
                     for visit in range(n)}

    tvisitInterval = {(visit, veh): mdl.interval_var(optional=True, name=f'tvisitInterval_{visit}_{veh}')
                      for visit in range(n) for veh in range(q)}
                      # if visit != firstDepotVisit and visit != n}
    route = {veh: mdl.sequence_var([tvisitInterval[visit, veh] for visit in range(n)],
                                   types=[visit for visit in range(n)],
                                   name=f'route_{veh}')
             for veh in range(q)}
    truck = {veh: mdl.interval_var(optional=True, name=f'truck_{veh}')
             for veh in range(q)}

    # TODO: Time constraints and capacity constraints
    # for v in clientVisits:
    #     mdl.add(mdl.end_before_end(visitInterval[v], tvisitInterval[v, veh])
    #             for veh in vehicles)
    #     mdl.add(mdl.startBeforeStart(visitInterval[v], tvisitInterval[v, veh])
    #             for veh in vehicles)
    #
    # for veh in vehicles:
    #     mdl.add(mdl.span(route[veh], [tvisitInterval[v, veh] for v in clientVisits], Dist))
    #     mdl.add(mdl.NoOverlap(route[veh], Dist))
    #     mdl.add(mdl.start_at_start(tvisitInterval[firstDepotVisit, veh], truck[veh]))

    # mdl.add(mdl.maximize(mdl.static_lex(mdl.end_of(tvisitInterval[n][veh] for veh in range(q)),
    #                                     mdl.sum(mdl.presence_of(truck[veh]) for veh in range(q)))))

    mdl.add(mdl.span(visitInterval[0][veh], data.service_times[0],
                     tvisitInterval[0][veh], name='depot_time_window')
            for veh in range(q))

    mdl.add(mdl.no_overlap(route[veh], mdl.transition_distance_matrix(dist, tvisitInterval, veh),
                           name=f'travel_time_{veh}')
            for veh in range(q))

    mdl.add(mdl.last(route[veh], tvisitInterval[n][veh],
                     mdl.transition_distance_matrix(dist, tvisitInterval, veh), name='return_to_depot')
            for veh in range(q))

    mdl.add(mdl.alternative(visitInterval[v], [tvisitInterval[v, t] for t in range(q)], name=f'visit_{v}')
            for v in range(1, n))

    mdl.add(mdl.end_of(tvisitInterval[n][veh] for veh in range(q)))

    mdl.parameters.timelimit = 15  # Add running time limit

    return mdl


if __name__ == '__main__':
    main()
