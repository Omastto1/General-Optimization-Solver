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
    vrp = VRP(cvrp_prob)
    num_cust = vrp.get_num_customers()
    num_vehicles = vrp.get_num_vehicles()
    n = vrp.get_num_visits()

    mdl = CpoModel()

    # Prev variables, circuit, first/last
    prev = [mdl.integer_var(0, n - 1, "P{}".format(i)) for i in range(n)]
    for v, fv, lv in vrp.vehicles():
        mdl.add(prev[fv] == vrp.get_last((v - 1) % num_vehicles))

    before = vrp.customers() + vrp.first()
    for c in vrp.customers():
        mdl.add(mdl.allowed_assignments(prev[c], before))
        mdl.add(prev[c] != c)

    for _, fv, lv in vrp.vehicles():
        mdl.add(mdl.allowed_assignments(prev[lv], vrp.customers() + (fv,)))

    mdl.add(mdl.sub_circuit(prev))

    # Vehicle
    veh = [mdl.integer_var(0, num_vehicles - 1, "V{}".format(i)) for i in range(n)]
    for v, fv, lv in vrp.vehicles():
        mdl.add(veh[fv] == v)
        mdl.add(veh[lv] == v)
        mdl.add(mdl.element(veh, prev[lv]) == v)
    for c in vrp.customers():
        mdl.add(veh[c] == mdl.element(veh, prev[c]))

    # Demand
    load = [mdl.integer_var(0, vrp.get_capacity(), "L{}".format(i)) for i in range(num_vehicles)]
    used = mdl.integer_var(0, num_vehicles, 'U')
    cust_veh = [veh[c] for c in vrp.customers()]
    demand = [vrp.get_demand(c) for c in vrp.customers()]
    mdl.add(mdl.pack(load, cust_veh, demand, used))

    # Time
    start_time = [mdl.integer_var(vrp.get_earliest_start(i), vrp.get_latest_start(i), "T{}".format(i)) for i in
                  range(n)]
    for fv in vrp.first():
        mdl.add(start_time[fv] == 0)
    for i in vrp.customers() + vrp.last():
        arrive = mdl.element([start_time[j] + vrp.get_service_time(j) + vrp.get_distance(j, i) for j in range(n)],
                             prev[i])
        mdl.add(start_time[i] == mdl.max(arrive, vrp.get_earliest_start(i)))

    # Distance
    all_dist = []
    for i in vrp.customers() + vrp.last():
        ldist = [vrp.get_distance(j, i) for j in range(n)]
        all_dist.append(mdl.element(ldist, prev[i]))
    total_distance = mdl.sum(all_dist) / TIME_FACTOR

    # Variables with inferred values
    mdl.add(mdl.inferred(cust_veh + load + [used] + start_time))

    # Objective
    mdl.add(mdl.minimize(total_distance))

    # KPIs
    mdl.add_kpi(used, 'Used')

    data.vrp = vrp
    data.prev = prev
    data.veh = veh
    data.load = load
    data.start_time = start_time

    return mdl, data


def display_path(sol, data):
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
    # TODO: add route
    # get time
    time = sol.solver_infos['TotalTime']
    # create dictionary
    solution_dict = {'solution_name': solution_name, 'solution': solution_value, 'time': time}
    # save to json
    print(f'Saving solution to {fout}')
    with open(fout, 'w') as f:
        json.dump(solution_dict, f)


class Integer_model(Solver):
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
        display_path(self.solution, self.data)

    def validate_solution(self):
        validate_path(self.solution, self.data)

    def visualize_solution(self):
        visualize_path(self.solution, self.data, self.data)

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
    fname = os.path.dirname(os.path.abspath(__file__)) + "\\..\\..\\data\\VRPTW\\solomon_25\\C101.json"
    # fname = '/home/lukesmi1/General-Optimization-Solver/data/VRPTW/solomon_25/C101.json'
    fout = None
    tlim = 5
    # tlim = None

    if len(sys.argv) != 1:
        if len(sys.argv) >= 2:
            fname = sys.argv[1]
        if len(sys.argv) >= 3:
            fout = sys.argv[2]
        if len(sys.argv) >= 4:
            tlim = int(sys.argv[3])
        elif len(sys.argv) >= 5:
            print(f'Usage: {sys.argv[0]} <input file> <output folder path> <time limit>')
            print(f'len(sys.argv)={len(sys.argv)}')
            print(sys.argv)
            exit(1)

    print(f'input file={fname} output folder path={fout} time limit={tlim}s')

    with open(fname, 'r') as f:
        instace = json.load(f)

    cvrptw_prob = CVRPTWProblem()
    # cvrptw_prob.read(fname)
    cvrptw_prob.from_dict(instace['data'])
    model, data_model = build_model(cvrptw_prob)
    # solution = model.solve(TimeLimit=tlim,
    #                         agent='local',
    #                        execfile='/home/lukesmi1/Cplex/cpoptimizer/bin/x86-64_linux/cpoptimizer')
    solution = model.solve(TimeLimit=tlim)
    if solution:
        # display_solution(solution, data_model)
        # visualize_solution(solution, data_model, cvrptw_prob)
        validate_path(solution, data_model)
        if not fout:
            head, tail = os.path.split(fname)
            tail = tail.split('.')[0] + '.json'
            fout = os.path.join(head, tail)
        save_solution(solution, fname, fout)
    else:
        print(f"No solution found for {fname}")
