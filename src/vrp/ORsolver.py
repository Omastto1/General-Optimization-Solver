import json
from math import sqrt

import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from pydantic import BaseModel

from Solver import *
from utils import *


class SolverSetting(BaseModel):
    time_limit: int


class Solver:
    """
    Solver object that takes a problem instance as input, creates and solves a capacitated vehicle routing problem with time
    windows. Objective of the optimization are hierarchical: 1) Minimize number of vehicles 2) Minimize total distance.
    Distance is Euclidean, and the value of travel time is equal to the value of distance between two nodes.

    Parameters
    ----------
    data : ProblemInstance
        Problem data according to ProblemInstance model.
    time_precision_scaler : int
        Variable defining the precision of travel and service times, e.g. 100 means precision of two decimals.
    """

    def __init__(self, time_precision_scaler: int):
        self.time = None
        self.time_precision_scaler = time_precision_scaler
        self.manager = None
        self.routing = None
        self.solution = None

    def create_model(self):
        """
        Create vehicle routing model for Solomon instance.
        """
        # Create the routing index manager, i.e. number of nodes, vehicles and depot
        self.manager = pywrapcp.RoutingIndexManager(
            len(self.data["time_matrix"]), self.data["num_vehicles"], self.data["depot"]
        )

        # Create routing model
        self.routing = pywrapcp.RoutingModel(self.manager)

        # Create and register a transit callback
        def time_callback(from_index, to_index):
            """Returns the travel time between the two nodes."""
            # Convert from solver internal routing variable Index to time matrix NodeIndex.
            from_node = self.manager.IndexToNode(from_index)
            to_node = self.manager.IndexToNode(to_index)
            return self.data["time_matrix"][from_node][to_node]

        transit_callback_index = self.routing.RegisterTransitCallback(time_callback)

        # Define cost of each arc and fixed vehicle cost
        self.routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        # Make sure to first minimize number of vehicles
        self.routing.SetFixedCostOfAllVehicles(100000)

        # Create and register demand callback
        def demand_callback(from_index):
            """Returns the demand of the node."""
            # Convert from routing variable Index to demands NodeIndex.
            from_node = self.manager.IndexToNode(from_index)
            return self.data["demands"][from_node]

        demand_callback_index = self.routing.RegisterUnaryTransitCallback(
            demand_callback
        )

        # Register vehicle capacitites
        self.routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,  # null capacity slack
            self.data["vehicle_capacities"],  # vehicle maximum capacities
            True,  # start cumul to zero
            "Capacity",
        )

        # Add Time Windows constraint.
        self.routing.AddDimension(
            transit_callback_index,
            10 ** 10,  # allow waiting time at nodes
            10 ** 10,  # maximum time per vehicle route
            False,  # Don't force start cumul to zero, i.e. vehicles can start after time 0 from depot
            "Time",
        )

        time_dimension = self.routing.GetDimensionOrDie("Time")

        # Add time window constraints for each location except depot.
        for location_idx, time_window in enumerate(self.data["time_windows"]):
            if location_idx == self.data["depot"]:
                continue
            index = self.manager.NodeToIndex(location_idx)
            time_dimension.CumulVar(index).SetRange(time_window[0], time_window[1])

        # Add time window constraints for each vehicle start node.
        depot_idx = self.data["depot"]
        for vehicle_id in range(self.data["num_vehicles"]):
            index = self.routing.Start(vehicle_id)
            time_dimension.CumulVar(index).SetRange(
                self.data["time_windows"][depot_idx][0],
                self.data["time_windows"][depot_idx][1],
            )
        # The solution finalizer is called each time a solution is found during search
        # and tries to optimize (min/max) variables values
        for i in range(self.data["num_vehicles"]):
            self.routing.AddVariableMinimizedByFinalizer(
                time_dimension.CumulVar(self.routing.Start(i))
            )
            self.routing.AddVariableMinimizedByFinalizer(
                time_dimension.CumulVar(self.routing.End(i))
            )

    def solve_model(self, settings: SolverSetting):
        """
        Solver model with solver settings.

        Parameters
        ----------
        settings : SolverSetting
            Solver settings according to SolverSetting model.
        """

        self.time = settings["time_limit"]
        # Setting first solution heuristic.
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
        )
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        search_parameters.time_limit.seconds = settings["time_limit"]

        # Solve the problem.
        self.solution = self.routing.SolveWithParameters(search_parameters)

    def print_solution(self):
        """
        Print solution to console.
        """
        print(f"Solution status: {self.routing.status()}\n")
        if self.routing.status() == 1:
            print(
                f"Objective: {self.solution.ObjectiveValue()/self.time_precision_scaler}\n"
            )
            time_dimension = self.routing.GetDimensionOrDie("Time")
            cap_dimension = self.routing.GetDimensionOrDie("Capacity")
            total_time = 0
            total_vehicles = 0
            for vehicle_id in range(self.data["num_vehicles"]):
                index = self.routing.Start(vehicle_id)
                plan_output = f"Route for vehicle {vehicle_id}:\n"
                while not self.routing.IsEnd(index):
                    time_var = time_dimension.CumulVar(index)
                    cap_var = cap_dimension.CumulVar(index)
                    plan_output += f"{self.manager.IndexToNode(index)} -> "
                    index = self.solution.Value(self.routing.NextVar(index))
                time_var = time_dimension.CumulVar(index)
                plan_output += f"{self.manager.IndexToNode(index)}\n"
                plan_output += f"Time of the route: {self.solution.Min(time_var)/self.time_precision_scaler}min\n"
                plan_output += f"Load of vehicle: {self.solution.Min(cap_var)}\n"
                total_time += self.solution.Min(time_var) / self.time_precision_scaler
                if self.solution.Min(time_var) > 0:
                    print(plan_output)
                    total_vehicles += 1
            total_travel_time = (
                total_time
                - sum(self.data["service_times"]) / self.time_precision_scaler
            )
            print(f"Total time of all routes: {total_time}min")
            print(f"Total travel time of all routes: {total_travel_time}min")
            print(f"Total vehicles used: {total_vehicles}")

    def get_solution(self):
        distance = 0
        total_vehicles = 0
        time = self.time
        solver = 'OR-Tools'
        paths = []

        if self.routing.status() == 1:
            # print(
            #     f"Objective: {self.solution.ObjectiveValue()/self.time_precision_scaler}\n"
            # )
            time_dimension = self.routing.GetDimensionOrDie("Time")
            cap_dimension = self.routing.GetDimensionOrDie("Capacity")
            total_vehicles = 0
            for vehicle_id in range(self.data["num_vehicles"]):
                index = self.routing.Start(vehicle_id)
                # plan_output = f"Route for vehicle {vehicle_id}:\n"
                path = [index]
                while not self.routing.IsEnd(index):
                    time_var = time_dimension.CumulVar(index)
                    cap_var = cap_dimension.CumulVar(index)
                    # plan_output += f"{self.manager.IndexToNode(index)} -> "
                    index = self.solution.Value(self.routing.NextVar(index))
                    path.append(index)
                time_var = time_dimension.CumulVar(index)
                # plan_output += f"{self.manager.IndexToNode(index)}\n"
                # plan_output += f"Time of the route: {self.solution.Min(time_var)/self.time_precision_scaler}min\n"
                # plan_output += f"Load of vehicle: {self.solution.Min(cap_var)}\n"
                distance += self.solution.Min(time_var) / self.time_precision_scaler
                if self.solution.Min(time_var) > 0:
                    # print(plan_output)
                    total_vehicles += 1
                    path = [i if i < len(self.data["service_times"]) else 0 for i in path]
                    paths.append(path)
            total_travel_time = (
                distance
                - sum(self.data["service_times"]) / self.time_precision_scaler
            )
            # print(f"Total time of all routes: {total_time}min")
            # print(f"Total travel time of all routes: {total_travel_time}min")
            # print(f"Total vehicles used: {total_vehicles}")

        return {'distance': 0, 'vehicles': total_vehicles, 'time': time, 'solver': solver, 'paths': paths}


    def load_instance(self, instance: CVRPTWProblem):
        """
        Load instance of Solomon benchmark with defined precision scaler.

        Parameters
        ----------
        time_precision_scaler : int
            Variable defining the precision of travel and service times, e.g. 100 means precision of two decimals.
            :param problem_path:
        """

        data = {}
        data["depot"] = 0
        data["service_times"] = [0] + [i * self.time_precision_scaler for i in instance.service_time]
        # data["ready_time"] = [0] + instance.earliest_start
        # data["due_date"] = [instance.max_horizon] + instance.latest_start

        dist = lambda p1, p2: sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))
        data["time_matrix"] = np.zeros((instance.nb_customers + 1, instance.nb_customers + 1))

        for i in range(instance.nb_customers + 1):
            for j in range(instance.nb_customers + 1):
                if i == j:
                    data["time_matrix"][i][j] = 0
                else:
                    data["time_matrix"][i][j] = int(dist(instance.xy[i], instance.xy[j]) * self.time_precision_scaler) + data["service_times"][j]

        data["time_matrix"] = data["time_matrix"].astype(int).tolist()

        data["demands"] = [0] + instance.demands

        time_windows = []
        for i in range(instance.nb_customers):
            time_windows.append(
                (
                    instance.earliest_start[i] * self.time_precision_scaler + instance.service_time[i] * self.time_precision_scaler,
                    instance.latest_start[i] * self.time_precision_scaler + instance.service_time[i] * self.time_precision_scaler,
                )
            )
        data["time_windows"] = [(0, instance.max_horizon * self.time_precision_scaler)] + time_windows
        data["num_vehicles"] = instance.nb_trucks
        data["vehicle_capacities"] = [instance.truck_capacity for _ in range(instance.nb_trucks)]

        self.data = data


def path_to_distance(path, data):
    vrp = data

    total_distance = 0
    for route in path['paths']:
        if len(route) > 2:
            for idx, nd in enumerate(route[:-1]):
                nxt = route[idx + 1]
                locald = vrp.get_distance(nd, nxt)

                total_distance += locald

    total_distance /= TIME_FACTOR
    return total_distance


class ORsolver(Solver):
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
        with open(fname) as f:
            self.instance = json.load(f)
        self.data.from_dict(self.instance['data'])
        self.model = Solver(TIME_FACTOR)
        self.model.load_instance(self.data)
        self.model.create_model()

    def save_to_json(self, fout=None):
        if fout is None:
            fout = self.fname
        with open(fout, 'w') as f:
            json.dump(self.instance, f)

    def solve(self, tlim):
        settings = {'time_limit': tlim}
        self.model.solve_model(settings)
        # solver.print_solution()
        self.solution = self.model.get_solution()
        self.solution['total_distance'] = path_to_distance(self.solution, self.data)
        self.instance['solutions'].append(self.solution)

    def display_solution(self):
        self.model.print_solution()

    def validate_solution(self):
        validate_path(self.solution, self.data)

    def visualize_solution(self):
        visualize_path(self.solution, self.data, self.data)
