from copy import deepcopy

import numpy as np
from pymoo.algorithms.soo.nonconvex.pso import PSO

from pymoo.core.problem import ElementwiseProblem, Problem
from pymoo.optimize import minimize

# from src.rcpsp.solvers.solver_cp import RCPSPCPSolver
# from src.rcpsp.solvers.solver_ga import RCPSPGASolver
from src.general_optimization_solver import load_raw_instance, load_instance, load_raw_benchmark
from src.common.solver import GASolver
from src.vrp.problem import *

from queue import PriorityQueue


def decode_chromosome_fast(instance, chromosome):
    """
    Args:
        instance (_type_): _description_
        chromosome (_type_): _description_

    Returns:
        _type_: _description_
    """
    routes = []
    remaining_capacity = instance.get_capacity()
    current_time = 0
    route = [0]
    length = 0

    for customer in chromosome:

        # Check if the current vehicle can serve the customer
        if (remaining_capacity - instance.get_demand(customer) >= 0 and
                current_time + instance.get_distance(route[-1], customer) <= instance.get_latest_start(customer)):
            # Update remaining capacity and current time
            remaining_capacity -= instance.get_demand(customer)
            current_time = (max(current_time + instance.get_distance(route[-1], customer),
                                instance.get_earliest_start(customer)) + instance.get_service_time(customer))
            length += instance.get_distance(route[-1], customer)

            route.append(customer)
        else:
            # Start a new route for the next vehicle
            length += instance.get_distance(route[-1], 0) + instance.get_distance(0, customer)
            route.append(0)
            routes.append(route)
            remaining_capacity = instance.get_capacity() - instance.get_demand(customer)
            current_time = (max(instance.get_distance(0, customer), instance.get_earliest_start(customer))
                            + instance.get_service_time(customer))
            route = [0, customer]

    length += instance.get_distance(route[-1], 0)
    route.append(0)
    routes.append(route)

    return routes, length


def decode_chromosome_rec(instance, chromosome):
    """
    Use recursive function to decode the chromosome and find more solutions
    Args:
        instance (_type_): _description_
        chromosome (_type_): _description_

    Returns:
        _type_: _description_
    """

    def recursive_decode(idx, routes1, route1, remaining_capacity1, current_time1, length1):
        if idx == len(chromosome):
            length1 += instance.get_distance(route1[-1], 0)
            route1.append(0)
            routes1.append(route1)
            return routes1, length1

        customer = chromosome[idx]

        # Find paths if we use a new vehicle
        routes2 = routes1.copy()
        route2 = route1.copy()

        route2.append(0)
        routes2.append(route2)
        remaining_capacity2 = instance.truck_capacity
        current_time2 = 0
        route2 = [0, customer]
        length2 = length1 + instance.get_distance(route1[-1], 0) + instance.get_distance(0, customer)

        second, new_length2 = recursive_decode(idx + 1, routes2, route2, remaining_capacity2, current_time2, length2)

        # Check if the current vehicle can serve the customer
        if (remaining_capacity1 >= instance.get_demand(customer) and
                current_time1 + instance.get_distance(route1[-1], customer) <= instance.get_latest_start(customer)):
            # Check the branch where the customer is served by the current vehicle
            time = (max(current_time1 + instance.get_distance(route1[-1], customer),
                        instance.get_earliest_start(customer))
                    + instance.get_service_time(customer))

            first, new_length1 = recursive_decode(idx + 1, routes1.copy(), route1 + [customer], remaining_capacity1 -
                                                  instance.get_demand(customer), time,
                                                  length1 + instance.get_distance(route1[-1], customer))
        else:
            return second, new_length2

        # Find the best solution
        if new_length1 <= new_length2:
            return first, new_length1
        else:
            return second, new_length2

    routes = []
    remaining_capacity = instance.truck_capacity
    current_time = 0
    route = [0]
    length = 0

    routes, length = recursive_decode(0, routes, route, remaining_capacity, current_time, length)

    return routes


def decode_chromosome_rec_pruned(instance, chromosome):
    def recursive_decode(idx, routes1, route1, remaining_capacity1, current_time1, length1):
        if idx == len(chromosome):
            if route1[-1] != 0:
                length1 += instance.get_distance(route1[-1], 0)
                route1.append(0)
            routes1.append(route1)
            return routes1, length1

        customer = chromosome[idx]

        # Find paths if we use a new vehicle
        routes2 = routes1.copy()
        route2 = route1.copy()

        route2.append(0)
        routes2.append(route2)
        remaining_capacity2 = instance.get_capacity() - instance.get_demand(customer)
        current_time2 = (max(instance.get_distance(0, customer), instance.get_earliest_start(customer))
                         + instance.get_service_time(customer))
        route2 = [0, customer]
        length2 = length1 + instance.get_distance(route1[-1], 0) + instance.get_distance(0, customer)

        # Check if the current vehicle can serve the customer
        if idx >= len(chromosome) - 1 or (
                remaining_capacity1 - instance.get_demand(customer) >= 0 and
                current_time1 + instance.get_distance(route1[-1], customer) <= instance.get_latest_start(customer)):
            # Check the branch where the customer is served by the current vehicle
            time = (max(current_time1 + instance.get_distance(route1[-1], customer),
                        instance.get_earliest_start(customer))
                    + instance.get_service_time(customer))

            first, new_length1 = recursive_decode(idx + 1, routes1.copy(), route1 + [customer], remaining_capacity1 -
                                                  instance.get_demand(customer), time,
                                                  length1 + instance.get_distance(route1[-1], customer))
        else:
            second, new_length2 = recursive_decode(idx + 1, routes2, route2, remaining_capacity2, current_time2,
                                                   length2)
            return second, new_length2

        if (instance.get_distance(chromosome[idx - 1], 0) / 2 <= instance.get_distance(customer, chromosome[idx - 1])
            and instance.get_distance(0, customer) / 2 <= instance.get_distance(customer, chromosome[idx - 1])) \
                or remaining_capacity1 - instance.get_demand(customer) < instance.get_demand(chromosome[idx + 1]):
            # print("Exploring", customer)
            second, new_length2 = recursive_decode(idx + 1, routes2, route2, remaining_capacity2, current_time2,
                                                   length2)
        else:
            return first, new_length1

        # Find the best solution
        if new_length1 <= new_length2:
            return first, new_length1
        else:
            return second, new_length2

    chromosome.append(0)
    routes = []
    remaining_capacity = instance.truck_capacity
    current_time = 0
    route = [0]
    length = 0

    routes, length = recursive_decode(0, routes, route, remaining_capacity, current_time, length)

    return routes, length


def decode_chromosome_rec_pruned_less(instance, chromosome):
    def recursive_decode(idx, routes1, route1, remaining_capacity1, current_time1, length1):
        if idx == len(chromosome):
            if route1[-1] != 0:
                length1 += instance.get_distance(route1[-1], 0)
                route1.append(0)
            routes1.append(route1)
            return routes1, length1

        customer = chromosome[idx]

        # Find paths if we use a new vehicle
        routes2 = routes1.copy()
        route2 = route1.copy()

        route2.append(0)
        routes2.append(route2)
        remaining_capacity2 = instance.get_capacity() - instance.get_demand(customer)
        current_time2 = (max(instance.get_distance(0, customer), instance.get_earliest_start(customer))
                         + instance.get_service_time(customer))
        route2 = [0, customer]
        length2 = length1 + instance.get_distance(route1[-1], 0) + instance.get_distance(0, customer)

        # print("Chromosome", chromosome)
        # Check if the current vehicle can serve the customer
        if idx >= len(chromosome) - 1 or (
                remaining_capacity1 - instance.get_demand(customer) >= 0 and
                current_time1 + instance.get_distance(route1[-1], customer) <= instance.get_latest_start(customer)):
            # Check the branch where the customer is served by the current vehicle
            time = (max(current_time1 + instance.get_distance(route1[-1], customer),
                        instance.get_earliest_start(customer))
                    + instance.get_service_time(customer))

            first, new_length1 = recursive_decode(idx + 1, routes1.copy(), route1 + [customer], remaining_capacity1 -
                                                  instance.get_demand(customer), time,
                                                  length1 + instance.get_distance(route1[-1], customer))
        else:
            second, new_length2 = recursive_decode(idx + 1, routes2, route2, remaining_capacity2, current_time2,
                                                   length2)
            return second, new_length2

        if (instance.get_distance(chromosome[idx - 1], 0) <= instance.get_distance(customer, chromosome[idx - 1])
            and instance.get_distance(0, customer) <= instance.get_distance(customer, chromosome[idx - 1])) \
                or remaining_capacity1 - instance.get_demand(customer) < instance.get_demand(chromosome[idx + 1]):
            # print("Exploring", customer)
            second, new_length2 = recursive_decode(idx + 1, routes2, route2, remaining_capacity2, current_time2,
                                                   length2)
        else:
            return first, new_length1

        # Find the best solution
        if new_length1 <= new_length2:
            if (instance.get_distance(chromosome[idx - 1], 0) <= instance.get_distance(customer, chromosome[idx - 1])
                    and instance.get_distance(0, customer) <= instance.get_distance(customer, chromosome[idx - 1])):
                print(f"Using old vehicle because of distance: {instance.get_distance(chromosome[idx - 1], 0)} at {idx}")
            else:
                print(f"Using old vehicle because of demand: {remaining_capacity1 - instance.get_demand(customer)} at {idx}")
            return first, new_length1
        else:
            if (instance.get_distance(chromosome[idx - 1], 0) <= instance.get_distance(customer, chromosome[idx - 1])
                    and instance.get_distance(0, customer) <= instance.get_distance(customer, chromosome[idx - 1])):
                print(f"Using new vehicle because of distance: {instance.get_distance(chromosome[idx - 1], 0)} at {idx}")
            else:
                print(f"Using new vehicle because of demand: {remaining_capacity1 - instance.get_demand(customer)} at {idx}")
            return second, new_length2

    # print("Chromosome", chromosome)
    chromosome.append(0)
    routes = []
    remaining_capacity = instance.truck_capacity
    current_time = 0
    route = [0]
    length = 0

    routes, length = recursive_decode(0, routes, route, remaining_capacity, current_time, length)

    return routes, length


def decode_chromosome_rec_pruned_less2(instance, chromosome):
    def recursive_decode(idx, routes1, route1, remaining_capacity1, current_time1, length1):
        if idx == len(chromosome):
            if route1[-1] != 0:
                length1 += instance.get_distance(route1[-1], 0)
                route1.append(0)
            routes1.append(route1)
            return routes1, length1

        customer = chromosome[idx]

        # Find paths if we use a new vehicle
        routes2 = routes1.copy()
        route2 = route1.copy()

        route2.append(0)
        routes2.append(route2)
        remaining_capacity2 = instance.get_capacity() - instance.get_demand(customer)
        current_time2 = (max(instance.get_distance(0, customer), instance.get_earliest_start(customer))
                         + instance.get_service_time(customer))
        route2 = [0, customer]
        length2 = length1 + instance.get_distance(route1[-1], 0) + instance.get_distance(0, customer)

        # Check if the current vehicle can serve the customer
        if idx >= len(chromosome) - 1 or (
                remaining_capacity1 - instance.get_demand(customer) >= 0 and
                current_time1 + instance.get_distance(route1[-1], customer) <= instance.get_latest_start(customer)):
            # Check the branch where the customer is served by the current vehicle
            time = (max(current_time1 + instance.get_distance(route1[-1], customer),
                        instance.get_earliest_start(customer))
                    + instance.get_service_time(customer))

            first, new_length1 = recursive_decode(idx + 1, routes1.copy(), route1 + [customer], remaining_capacity1 -
                                                  instance.get_demand(customer), time,
                                                  length1 + instance.get_distance(route1[-1], customer))
        else:
            second, new_length2 = recursive_decode(idx + 1, routes2, route2, remaining_capacity2, current_time2,
                                                   length2)
            return second, new_length2

        if (instance.get_distance(chromosome[idx - 1], 0) <= instance.get_distance(customer, chromosome[idx - 1])
                and instance.get_distance(0, customer) <= instance.get_distance(customer, chromosome[idx - 1])):
            # print("Exploring", customer)
            second, new_length2 = recursive_decode(idx + 1, routes2, route2, remaining_capacity2, current_time2,
                                                   length2)
        else:
            return first, new_length1

        # Find the best solution
        if new_length1 <= new_length2:
            return first, new_length1
        else:
            return second, new_length2

    chromosome.append(0)
    routes = []
    remaining_capacity = instance.truck_capacity
    current_time = 0
    route = [0]
    length = 0

    routes, length = recursive_decode(0, routes, route, remaining_capacity, current_time, length)

    return routes, length


def decode_chromosome_second(instance, chromosome):
    """
    Args:
        instance (_type_): _description_
        chromosome (_type_): _description_

    Returns:
        _type_: _description_
    """
    routes = []
    remaining_capacity = instance.get_capacity()
    current_time = 0
    route = [0]
    length = 0

    for customer in chromosome:
        c_to_next = instance.get_distance(route[-1], customer)
        if (instance.get_distance(route[-1], 0) <= c_to_next
                and instance.get_distance(0, customer) <= c_to_next):
            # Start a new route for the next vehicle
            length += instance.get_distance(route[-1], 0) + instance.get_distance(0, customer)
            route.append(0)
            routes.append(route)
            remaining_capacity = instance.get_capacity() - instance.get_demand(customer)
            current_time = (max(instance.get_distance(0, customer), instance.get_earliest_start(customer))
                            + instance.get_service_time(customer))
            route = [0, customer]
        elif (remaining_capacity - instance.get_demand(customer) >= 0 and
                current_time + c_to_next <= instance.get_latest_start(customer)):

            # Update remaining capacity and current time
            remaining_capacity -= instance.get_demand(customer)
            current_time = (max(current_time + c_to_next,
                                instance.get_earliest_start(customer)) + instance.get_service_time(customer))
            length += c_to_next

            route.append(customer)
        else:
            # Start a new route for the next vehicle
            length += instance.get_distance(route[-1], 0) + instance.get_distance(0, customer)
            route.append(0)
            routes.append(route)
            remaining_capacity = instance.get_capacity() - instance.get_demand(customer)
            current_time = (max(instance.get_distance(0, customer), instance.get_earliest_start(customer))
                            + instance.get_service_time(customer))
            route = [0, customer]

    length += instance.get_distance(route[-1], 0)
    route.append(0)
    routes.append(route)

    return routes, length


def decode_chromosome_smart(instance, chromosome):
    # TODO: This decoder would precompute all distances for given order of customers and solve it like the first decoder only using a new vehicle when it has to. It would then only move the points at which a new car is used to the left based on the difference of distance to the next customer and to the depo and back. This might save a lot of computation time since not all routes after the change might need to be reevaluated.
    pass


def decode_chromosome_iterative_ordered(instance, chromosome):
    # TODO: The idea for this decoder is based on combining iterative deepening search with a search heuristic. It would compute the path until it has to use a new vehicle and then compute all the paths where new vehicle might be better until the point where the previous path stopped. There it would evaluate the best one and repeat from there.
    pass


def route_to_length(instance, routes):
    """
    Args:
        instance (_type_): _description_
        route (_type_): _description_

    Returns:
        _type_: _description_
    """

    total_length = 0
    for route in routes:
        for i in range(len(route) - 1):
            total_length += instance.get_distance(route[i], route[i + 1])

    return total_length


def fitness_func(instance, x, out):
    """Fitness func that is being fed to pymoo algorithm

    Args:
        instance (_type_): _description_
        x (list[float]): chromosome
        out (dict): pymoo specific output

    Returns:
        out (dict): pymoo specific output
    """

    # print("Chromosome shape", x.shape)
    # print("Chromosome", x)
    chromosome = sorted(range(1, len(x) + 1), key=lambda i: x[i - 1])
    # routes, dist = decode_chromosome_rec_pruned_less(instance, chromosome)
    routes, dist = decode_chromosome_second(instance, chromosome)

    # print(f"DISTANCE: {dist}")

    out["F"] = dist
    # out["routes"] = {0: routes}     # For some reason, pymoo requires routes to be in a structure, string and dicts work

    # print(out)

    return out


class VRPTWSolver(GASolver):
    """GA SOLVER WRAPPER CLASS
    """

    def _solve(self, instance, validate=False, visualize=False, force_execution=False, update_history=True):
        class VRPTW(ElementwiseProblem):
            """pymoo wrapper class
            """

            def __init__(self, instance, fitness_func_):
                # print("number of customers", instance.nb_customers)
                super().__init__(n_var=instance.nb_customers, n_obj=1, xl=0, xu=1)
                self.instance = instance
                self.fitness_func = fitness_func_

            def _evaluate(self, x, out, *args, **kwargs):
                out = self.fitness_func(self.instance, x, out)

                assert "solution" not in out, "Do not use `solution` key, it is pymoo reserved keyword"

                return out

        problem = VRPTW(instance, self.fitness_func)
        res = minimize(problem, self.algorithm, self.termination,
                       verbose=False, seed=self.seed,
                       callback=self.callback)

        if res.F is None:
            print("No solution found")
            return None, None, res

        chromosome = sorted(range(1, len(res.X) + 1), key=lambda i: res.X[i - 1])

        #   Take min from the 2 non-recursive decoders?
        routes, dist = decode_chromosome_second(instance, chromosome)

        fitness_value = dist/10

        # print(f"Fitness value: {fitness_value}"
        #       f"\nRoutes: {routes}")

        export = {'paths': routes, 'total_distance': fitness_value}

        if validate:
            try:
                print("Validating solution...")
                is_valid = validate_path(export, instance)
                if is_valid:
                    print("Solution is valid.")
                else:
                    print("Solution is invalid.")
            except AssertionError as e:
                print("Solution is invalid.")
                print(e)
                return None, export, res

        if visualize:
            visualize_path(export, instance)

        if update_history:
            fitness_value = fitness_value
            solution_info = {'total_distance': fitness_value, 'n_vehicles': len(routes), 'paths': routes}
            solution_progress = deepcopy(res.algorithm.callback.data['progress'])
            self.add_run_to_history(instance, fitness_value, solution_info, solution_progress,
                                    exec_time=round(res.exec_time, 2))

        return fitness_value, export, res


# if __name__ == "__main__":
#     path = "..\\..\\..\\data\\VRPTW\\solomon_25\\C101.json"
#
#     algorithm = PSO()
#     instance = load_instance(path)
#     solver = VRPTWSolver(algorithm=algorithm, fitness_func=fitness_func, termination=('n_gen', 100), seed=1, solver_name="PSO")
#
#     res = solver.solve(instance, validate=True, visualize=True)


if __name__ == "__main__":
    path = "..\\..\\..\\data\\VRPTW\\solomon_100\\C206.json"

    algorithm = PSO()
    instance = load_instance(path)
    solver = VRPTWSolver(algorithm=algorithm, fitness_func=fitness_func, termination=('n_gen', 1000), solver_name="GA")

    res = solver.solve(instance, validate=True, visualize=True)

    print(res)

    print(instance)

    instance.dump(verbose=True, dir_path="..\\..\\..\\data\\VRPTW")
