import datetime
import sys
import os
import math
import json
import matplotlib.pyplot as plt
from collections import namedtuple
import platform
import re
import multiprocessing

from docplex.cp.model import CpoModel, CpoParameters
import docplex.cp.solver.solver as solver
from docplex.cp.utils import compare_natural

from Solver import *
from utils import *

# print("solver.get_version_info()=", solver.get_version_info())
solver_version = solver.get_version_info()['SolverVersion']
if compare_natural(solver_version, '22.1.1.0') < 0:
    print('Warning solver version', solver_version, 'is too old for', __file__)
#     exit(0)

class VRP:
    VisitData = namedtuple("CustomerData", "demand service_time earliest, latest")

    def __init__(self, pb):
        # Sizes
        self._num_veh = pb.get_nb_trucks()
        self._num_cust = pb.get_num_nodes() - 1
        self._n = self._num_cust + self._num_veh * 2

        # First, last, customer groups
        self._first = tuple(self._num_cust + i for i in range(self._num_veh))
        self._last = tuple(self._num_cust + self._num_veh + i for i in range(self._num_veh))
        self._cust = tuple(range(self._num_cust))

        # Time and load limits
        self._max_horizon = pb.get_max_horizon()
        self._capacity = pb.get_capacity()

        # Node mapping
        pnode = [i + 1 for i in range(self._num_cust)] + [0] * (2 * self._num_veh)

        # Visit data
        self._visit_data = \
            tuple(VRP.VisitData(pb.get_demand(pnode[c]), pb.get_service_time(pnode[c]), pb.get_earliest_start(pnode[c]),
                                pb.get_latest_start(pnode[c])) for c in self._cust) + \
            tuple(VRP.VisitData(0, 0, 0, self._max_horizon) for _ in self._first + self._last)

        # Distance
        self.distance_matrix = [
            [pb.get_distance(pnode[i], pnode[j]) for j in range(self._n)]
            for i in range(self._n)
        ]

    def first(self): return self._first

    def last(self): return self._last

    def vehicles(self): return zip(range(self._num_veh), self._first, self._last)

    def customers(self): return self._cust

    def all(self): return range(self._n)

    def get_num_customers(self): return self._num_cust

    def get_num_visits(self): return self._n

    def get_num_vehicles(self): return self._num_veh

    def get_first(self, veh): return self._first[veh]

    def get_last(self, veh): return self._last[veh]

    def get_capacity(self): return self._capacity

    def get_max_horizon(self): return self._max_horizon

    def get_demand(self, i): return self._visit_data[i].demand

    def get_service_time(self, i): return self._visit_data[i].service_time

    def get_earliest_start(self, i): return self._visit_data[i].earliest

    def get_latest_start(self, i): return self._visit_data[i].latest

    def get_distance(self, i, j): return self.distance_matrix[i][j]


class DataModel:
    vrp = None
    prev = None
    veh = None
    load = None
    start_time = None
    params = None


def build_model(cvrp_prob):
    data = DataModel()
    vrp = VRP(cvrp_prob)  # remove VRP class
    num_cust = vrp.get_num_customers()
    num_vehicles = vrp.get_num_vehicles()
    n = vrp.get_num_visits()

    print("num_cust=", num_cust)
    print("num_vehicles=", num_vehicles)
    print("n=", n)

    mdl = CpoModel()

    # job_operations = [[model.interval_var(name=f"J_{job}_{order_index}", size=instance.durations[job][order_index])
    #                            for order_index in range(instance.no_machines)] for job in range(instance.no_jobs)]
    # interval variable that represents the time interval of size
    # h
    # l
    # i for the visit of vertex i 2 V

    visit = [mdl.interval_var(size=vrp.get_service_time(i), name="V{}".format(i)) for i in range(n)]

    #  dvar interval wtvisitInterval [v in clientVisits] size v.dropTime..horizon;

    twVisitInterval = [mdl.interval_var(start=vrp.get_earliest_start(i), end=vrp.get_latest_start(i) + vrp.get_service_time(i), name="TW{}".format(i)) for i in range(num_cust)]

    # optional interval variable that represents the time
    # interval for the visit of vehicle k 2 K l at vertex
    # i 2 Vþ

    visit_veh = [[mdl.interval_var(optional=True, name="V{}_{}".format(i, vehicle)) if i != num_cust + vehicle and i != num_cust + num_vehicles + vehicle else mdl.interval_var(name="V{}_{}".format(i, vehicle)) for i in range(n)] for vehicle in range(num_vehicles)]

    # dvar sequence route[veh in vehicles] in all(v in allVisits) tvisitInterval[v][veh]
    #                                   types all(v in allVisits) ord(allVisits,v);

    route = [mdl.sequence_var([visit_veh[vehicle][i] for i in range(n)], types=[i for i in range(n)], name="R{}".format(vehicle)) for vehicle in range(num_vehicles)]

    # dvar interval truck [veh in vehicles] optional;

    truck = [mdl.interval_var(optional=True, name="T{}".format(k)) for k in range(num_vehicles)]

    # dexpr float travelMaxTime = max(veh in vehicles) endOf(tvisitInterval[lastDepotVisit][veh]);

    travel_max_time = mdl.sum([mdl.end_of(visit_veh[vehicle][num_cust + num_vehicles + vehicle]) for vehicle in range(num_vehicles)])

    # dexpr int nbUsed = sum(veh in vehicles) presenceOf(truck[veh]);

    nb_used = mdl.sum([mdl.presence_of(truck[k]) for k in range(num_vehicles)])

    # dexpr int load[veh in vehicles] = sum(v in clientVisits) presenceOf(tvisitInterval[v][veh])*v.quantity;

    load = [mdl.sum([mdl.presence_of(visit_veh[vehicle][i]) * vrp.get_demand(i-1) for i in range(n)]) for vehicle in range(num_vehicles)]

    # minimize staticLex(travelMaxTime,nbUsed);

    # mdl.add(mdl.minimize_static_lex([travel_max_time, nb_used]))
    mdl.add(mdl.minimize(travel_max_time))

    # print("len(vrp.distance_matrix)= ", len(vrp.distance_matrix))
    # for i in range(len(vrp.distance_matrix)):
    #     print(f"vrp.distance_matrix[{i}]= ", vrp.distance_matrix[i])
    # print(vrp.distance_matrix)

    #   forall(veh in vehicles) {
    for vehicle in range(num_vehicles):
        #   	span (truck[veh], all(v in clientVisits) tvisitInterval[v][veh]);
        # mdl.add(mdl.span(truck[vehicle], [visit_veh[vehicle][i] for i in range(n)]))
        #     noOverlap(route[veh], Dist);          // Travel time
        mdl.add(mdl.no_overlap(route[vehicle], vrp.distance_matrix))
        #     startOf(tvisitInterval[firstDepotVisit][veh])==0;     // Truck t starts at time 0 from depot
        mdl.add(mdl.start_of(visit_veh[vehicle][num_cust + vehicle]) == 0)
        mdl.add(mdl.first(route[vehicle], visit_veh[vehicle][num_cust + vehicle]))
        #     last (route[veh],tvisitInterval[lastDepotVisit] [veh]); // Truck t returns at depot
        mdl.add(mdl.last(route[vehicle], visit_veh[vehicle][num_cust + num_vehicles + vehicle]))
        # print(f"Vehicle {vehicle} starts at {num_cust + vehicle} and ends at {num_cust + num_vehicles + vehicle}")
        # load[veh] <= veh.capacity;                       // Truck capacity
        # temp1 = load[vehicle]
        # temp2 = vrp.get_capacity()
        # print(f"temp1= {temp1}")
        # print()
        # print(f"temp2= {temp2}")
        mdl.add(load[vehicle] <= vrp.get_capacity())

    #   forall(v in clientVisits)
    #     alternative(visitInterval[v], all(t in vehicles) tvisitInterval[v][t]); // Truck selection
    for i in range(n):
        mdl.add(mdl.alternative(visit[i], [visit_veh[k][i] for k in range(num_vehicles)]))

    for i in range(num_cust):
        if i == 23:
            print(i, visit[i], twVisitInterval[i])
        # endAtEnd(wtvisitInterval[v], visitInterval[v]);
        mdl.add(mdl.end_before_end(visit[i], twVisitInterval[i], ))

        # mdl.add(mdl.end_at_end(twVisitInterval[i], visit[i]))
        # startBeforeStart(wtvisitInterval[v], visitInterval[v]);
        mdl.add(mdl.start_before_start(twVisitInterval[i], visit[i]))

    return mdl, data


def display_solution(sol, data):
    vrp = data

    v = 0
    for route in sol['paths']:
        if len(route) > 2:
            print('Veh {} --->'.format(v, route), end="")
            arrive = 0
            total_distance = 0
            total_load = 0
            line = ""
            for idx, nd in enumerate(route):
                early = vrp.get_earliest_start(nd)
                late = vrp.get_latest_start(nd)
                start = max(arrive, early)
                line += " {} (a = {}, t = {} <= {} <= {})".format(nd, arrive, early, start, late)
                if nd != route[-1]:
                    nxt = route[idx + 1]
                    locald = vrp.get_distance(nd, nxt)
                    serv = vrp.get_service_time(nd)
                    line += " -- {} + {} -->".format(serv, locald)
                    arrive = start + serv + locald
                    total_distance += locald
                    if nd != route[0]:
                        total_load += vrp.get_demand(nd)
            line += " --- D = {:.1f}, L = {}".format(total_distance, total_load)
            print(line)
        v += 1


def get_solution(sol, data):
    vrp = data.vrp
    sprev = tuple(sol.solution[p] for p in data.prev)

    n_vehicles = 0
    total_distance = 0
    paths = []

    for v, fv, lv in vrp.vehicles():
        route = []
        nd = lv
        while nd != fv:
            route.append(nd)
            nd = sprev[nd]
        route.append(fv)
        route.reverse()

        if len(route) > 2:
            n_vehicles += 1
            paths.append([0 if i + 1 > vrp.get_num_customers() else i + 1 for i in route])

            for idx, nd in enumerate(route):
                if nd != route[-1]:
                    nxt = route[idx + 1]
                    locald = vrp.get_distance(nd, nxt)
                    total_distance += locald

    total_distance /= TIME_FACTOR
    ret = {'n_vehicles': n_vehicles, 'total_distance': total_distance, 'paths': paths}
    return ret


def save_solution(sol, path_to_instance, fout):
    # save solution to json with name, solution and time

    # get instance name
    solution_name = '.'.join(path_to_instance.split('.')[:-1]) + '.json'
    # get solution
    solution_value = sol.solution.objective_values[0]
    # get time
    time = sol.solver_infos['TotalTime']
    # create dictionary
    solution_dict = {'solution_name': solution_name, 'solution': solution_value, 'time': time}
    # save to json
    print(f'Saving solution to {fout}')
    with open(fout, 'w') as f:
        json.dump(solution_dict, f)


class Interval_model(Solver):
    def __init__(self):
        self.sol = None
        self.solution = None
        self.fname = None
        self.model = None
        self.data_model = None
        self.instance = None

        self.data = CVRPTWProblem()

    def read_json(self, fname):
        self.fname = fname
        with open(fname, 'r') as f:
            self.instance = json.load(f)
        self.data.from_dict(self.instance['data'])
        self.model, self.data_model = build_model(self.data)

    def save_to_json(self, fout=None):
        if fout is None:
            fout = self.fname
        with open(fout, 'w') as f:
            json.dump(self.instance, f)

    def solve(self, tlim, workers=None, execfile='/home/lukesmi1/Cplex/cpoptimizer/bin/x86-64_linux/cpoptimizer'):
        # Solver params setting
        params = CpoParameters()
        # params.SearchType = 'Restart'
        params.LogPeriod = 100000
        params.LogVerbosity = 'Terse'
        if workers is not None:
            params.Workers = workers

        self.model.set_parameters(params=params)
        self.data_model.params = params

        if platform.system() == 'Windows' and execfile == '/home/lukesmi1/Cplex/cpoptimizer/bin/x86-64_linux/cpoptimizer':
            self.sol = self.model.solve(TimeLimit=tlim)
        else:
            self.sol = self.model.solve(TimeLimit=tlim, agent='local', execfile=execfile)
        # Get number of cars and their paths from solution
        self.solution = get_solution(self.sol, self.data_model)

        # self.validate_solution()

        # Add to solution time from solution, number of cores, solver version and current time
        self.solution['time'] = self.sol.solver_infos['TotalTime']
        self.solution['reason'] = self.sol.solver_infos['SearchStopCause']
        self.solution['n_workers'] = self.sol.solver_infos['EffectiveWorkers']
        self.solution['n_cores'] = multiprocessing.cpu_count()   # Doesn't work on cluster
        self.solution['solver_version'] = self.sol.process_infos['SolverVersion']
        self.solution['date'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        log = self.sol.solver_log

        # Define the regex pattern
        pattern = r"\*\s+(\d+\.\d+)\s+\w+\s+(\d+\.\d+s)"

        # Find all matches of numbers and times in the log using the regex pattern
        matches = re.findall(pattern, log, re.MULTILINE)

        # Convert minutes and hours into seconds and store the results
        result = [[float(match[0]), match[1]] for match in matches]
        for i in range(len(result)):
            unit = result[i][1][-1]  # Get the last character of the time
            if unit not in ['s', 'm', 'h']:  # If the unit is not minutes or hours
                print("Error: Unknown unit", unit)
            time = float(result[i][1][:-1])  # Get the time without the last character
            if unit == 'm':  # If the unit is minutes
                result[i][1] = time * 60  # Convert minutes to seconds
            elif unit == 'h':  # If the unit is hours
                result[i][1] = time * 3600  # Convert hours to seconds
            else:
                result[i][1] = time  # Otherwise, the unit is seconds

        self.solution['search_progress'] = result

        # Update instance with solution
        self.instance['solutions'].append(self.solution)

        if self.instance['our_best_solution'] == '' or self.instance['our_best_solution']['total_distance'] > self.solution['total_distance']:
            self.instance['our_best_solution'] = self.solution

    def display_solution(self):
        display_solution(self.solution, self.data)

    def validate_solution(self):
        validate_path(self.solution, self.data)

    def visualize_solution(self):
        visualize_path(self.solution, self.data, self.data)
        # visualize_solution(solution, data_model, cvrptw_prob)

    def visualize_progress(self, solution=None):
        if solution is None:
            solution = self.solution
        if solution is None:
            if len(self.instance['solutions']) == 0:
                print('No solution to visualize')
                return
            print('Visualizing last solution')
            solution = self.instance['solutions'][-1]

        numbers = [entry[0] for entry in solution['search_progress']]
        times = [entry[1] for entry in solution['search_progress']]

        best = self.instance['best_known_solution']['Distance']
        if best != '':
            best_known_solution = float(best)
            plt.axhline(y=best_known_solution, color='r', linestyle='--', label='Best Known Solution')
        if self.instance['our_best_solution']:
            our_best_solution = self.instance['our_best_solution']['total_distance']
            plt.axhline(y=our_best_solution, color='g', linestyle='--', label='Our Best Solution')

        # Plot the data
        plt.plot(times, numbers, 'bo-', label='Results')
        plt.ylabel('Value')
        plt.xlabel('Time (seconds)')
        plt.title('Results')
        plt.legend()

        # plt.ylim(min(times), max(times + [best_known_solution]))

        # Display the plot
        plt.show()


if __name__ == "__main__":
    pass
