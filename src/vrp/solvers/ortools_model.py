from src.common.solver import ORtoolsSolver
from src.vrp.problem import *

from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from pydantic import BaseModel

from math import sqrt
import numpy as np
import time as time


class SolverSetting(BaseModel):
    time_limit: int


class VRPTWSolver(ORtoolsSolver):
    solver_name = 'CP OR-tools Model'

    def build_model(self, instance):
        """
        Create vehicle routing model for Solomon instance.
        """
        data = self.load_instance(instance)
        # Create the routing index manager, i.e. number of nodes, vehicles and depot
        manager = pywrapcp.RoutingIndexManager(
            len(data["time_matrix"]), data["num_vehicles"], data["depot"]
        )

        # Create routing model
        routing = pywrapcp.RoutingModel(manager)

        class SolutionCallback(object):
            def __init__(self, model):
                self.model = model
                self.start_time = time.time()
                self.history = []

            def __call__(self):
                CurrTime = time.time()
                # print("Solution", CurrTime - self.start_time, self.model.CostVar().Max())
                self.history.append([self.model.CostVar().Max(), CurrTime - self.start_time])

        solution_callback = SolutionCallback(routing)
        routing.AddAtSolutionCallback(solution_callback)

        # Create and register a transit callback
        def time_callback(from_index, to_index):
            """Returns the travel time between the two nodes."""
            # Convert from solver internal routing variable Index to time matrix NodeIndex.
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return data["time_matrix"][from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(time_callback)

        # Define cost of each arc and fixed vehicle cost
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        # Make sure to first minimize number of vehicles
        routing.SetFixedCostOfAllVehicles(100000)

        # Create and register demand callback
        def demand_callback(from_index):
            """Returns the demand of the node."""
            # Convert from routing variable Index to demands NodeIndex.
            from_node = manager.IndexToNode(from_index)
            return data["demands"][from_node]

        demand_callback_index = routing.RegisterUnaryTransitCallback(
            demand_callback
        )

        # Register vehicle capacitites
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,  # null capacity slack
            data["vehicle_capacities"],  # vehicle maximum capacities
            True,  # start cumul to zero
            "Capacity",
        )

        # Add Time Windows constraint.
        routing.AddDimension(
            transit_callback_index,
            10 ** 10,  # allow waiting time at nodes
            10 ** 10,  # maximum time per vehicle route
            False,  # Don't force start cumul to zero, i.e. vehicles can start after time 0 from depot
            "Time",
        )

        time_dimension = routing.GetDimensionOrDie("Time")

        # Add time window constraints for each location except depot.
        for location_idx, time_window in enumerate(data["time_windows"]):
            if location_idx == data["depot"]:
                continue
            index = manager.NodeToIndex(location_idx)
            time_dimension.CumulVar(index).SetRange(time_window[0], time_window[1])

        # Add time window constraints for each vehicle start node.
        depot_idx = data["depot"]
        for vehicle_id in range(data["num_vehicles"]):
            index = routing.Start(vehicle_id)
            time_dimension.CumulVar(index).SetRange(
                data["time_windows"][depot_idx][0],
                data["time_windows"][depot_idx][1],
            )
        # The solution finalizer is called each time a solution is found during search
        # and tries to optimize (min/max) variables values
        for i in range(data["num_vehicles"]):
            routing.AddVariableMinimizedByFinalizer(
                time_dimension.CumulVar(routing.Start(i))
            )
            routing.AddVariableMinimizedByFinalizer(
                time_dimension.CumulVar(routing.End(i))
            )

        return routing, manager, solution_callback

    def load_instance(self, instance):
        """
        Load instance of Solomon benchmark with defined precision scaler.

        Parameters
        ----------
        time_precision_scaler : int
            Variable defining the precision of travel and service times, e.g. 100 means precision of two decimals.
            :param instance:
            :param problem_path:
        """

        data = {}
        data["depot"] = 0
        data["service_times"] = [0] + [i * TIME_FACTOR for i in instance.service_time]
        # data["ready_time"] = [0] + instance.earliest_start
        # data["due_date"] = [instance.max_horizon] + instance.latest_start

        dist = lambda p1, p2: sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))
        data["time_matrix"] = np.zeros((instance.nb_customers + 1, instance.nb_customers + 1))

        for i in range(instance.nb_customers + 1):
            for j in range(instance.nb_customers + 1):
                if i == j:
                    data["time_matrix"][i][j] = 0
                else:
                    data["time_matrix"][i][j] = int(dist(instance.xy[i], instance.xy[j]) * TIME_FACTOR) + data["service_times"][j]

        data["time_matrix"] = data["time_matrix"].astype(int).tolist()

        data["demands"] = [0] + instance.demands

        time_windows = []
        for i in range(instance.nb_customers):
            time_windows.append(
                (
                    instance.earliest_start[i] * TIME_FACTOR + instance.service_time[i] * TIME_FACTOR,
                    instance.latest_start[i] * TIME_FACTOR + instance.service_time[i] * TIME_FACTOR,
                )
            )
        data["time_windows"] = [(0, instance.max_horizon * TIME_FACTOR)] + time_windows
        data["num_vehicles"] = instance.nb_trucks
        data["vehicle_capacities"] = [instance.truck_capacity for _ in range(instance.nb_trucks)]

        return data

    def _export_solution(self, sol, data, routing):
        total_distance = 0
        total_vehicles = 0
        paths = []

        # print(
        #     f"Objective: {self.solution.ObjectiveValue()/TIME_FACTOR}\n"
        # )
        time_dimension = routing.GetDimensionOrDie("Time")
        cap_dimension = routing.GetDimensionOrDie("Capacity")

        for vehicle_id in range(data.nb_trucks):
            index = routing.Start(vehicle_id)
            # plan_output = f"Route for vehicle {vehicle_id}:\n"
            path = [index]
            while not routing.IsEnd(index):
                time_var = time_dimension.CumulVar(index)
                cap_var = cap_dimension.CumulVar(index)
                # plan_output += f"{self.manager.IndexToNode(index)} -> "
                index = sol.Value(routing.NextVar(index))
                path.append(index)
            time_var = time_dimension.CumulVar(index)
            # plan_output += f"{self.manager.IndexToNode(index)}\n"
            # plan_output += f"Time of the route: {self.solution.Min(time_var)/TIME_FACTOR}min\n"
            # plan_output += f"Load of vehicle: {self.solution.Min(cap_var)}\n"
            # total_distance += sol.Min(time_var) / TIME_FACTOR
            if sol.Min(time_var) > 0:
                # print(plan_output)
                total_vehicles += 1
                path = [i if i < data.nb_customers + 1 else 0 for i in path]
                for i in range(len(path) - 1):
                    total_distance += data.get_distance(path[i], path[i + 1])
                paths.append(path)
        # total_travel_time = (
        #         total_distance
        #         - sum(data["service_times"]) / TIME_FACTOR
        # )
        # print(f"Total time of all routes: {total_time}min")
        # print(f"Total travel time of all routes: {total_travel_time}min")
        # print(f"Total vehicles used: {total_vehicles}")

        # return {'total_distance': 0, 'vehicles': total_vehicles, 'time': time, 'solver': solver, 'paths': paths}

        total_distance = total_distance / TIME_FACTOR

        ret = {'n_vehicles': total_vehicles, 'total_distance': total_distance, 'paths': paths}
        return ret

    def _solve(self, instance, validate=False, visualize=False, force_execution=False, update_history=True):
        time_start = time.time()
        print("Building model")
        model, model_variables, history = self.build_model(instance)

        print("Looking for solution")

        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.time_limit.seconds = self.params.TimeLimit
        # search_parameters.log_search = True
        # solver.print_solution()

        sol = model.SolveWithParameters(search_parameters)

        if model.status() != 1 and model.status() != 2:
            print('No solution found')
            return None, None, sol

        result = self._export_solution(sol, instance, model)

        if validate:
            try:
                print("Validating solution...")
                is_valid = validate_path(result, instance)
                if is_valid:
                    print("Solution is valid.")
                else:
                    print("Solution is invalid.")
            except AssertionError as e:
                print("Solution is invalid.")
                print(e)
                return None, None, result

        if visualize:
            visualize_path(result, instance)

        obj_value = result['total_distance']
        print('Objective value:', obj_value)

        if model.status() == 1:
            print("Optimal solution found")
        elif model.status() == 2:
            print("Feasible solution found")
        else:
            print("Unknown solution status")
            print(model.status())

        instance.compare_to_reference(obj_value)

        time_end = time.time()

        if update_history:
            self.add_run_to_history(instance, sol, result, history.history, time_end - time_start)

        return obj_value, model_variables, sol
