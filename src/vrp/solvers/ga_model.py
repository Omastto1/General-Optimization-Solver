import numpy as np
import networkx as nx
from pymoo.algorithms.soo.nonconvex.pso import PSO

from pymoo.core.duplicate import ElementwiseDuplicateElimination
from pymoo.algorithms.soo.nonconvex.brkga import BRKGA
from pymoo.core.problem import ElementwiseProblem, Problem
from pymoo.optimize import minimize

# from src.rcpsp.solvers.solver_cp import RCPSPCPSolver
# from src.rcpsp.solvers.solver_ga import RCPSPGASolver
from src.general_optimization_solver import load_raw_instance, load_instance, load_raw_benchmark
from src.common.solver import GASolver


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
            return first, new_length1
        else:
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
    routes, dist = decode_chromosome_rec_pruned_less(instance, chromosome)

    # print(f"DISTANCE: {dist}")

    out["F"] = dist
    # out["routes"] = str(routes)

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
                super().__init__(n_var=instance.nb_customers, n_obj=1,
                                 n_ieq_constr=4, n_eq_constr=4, xl=0, xu=1)
                self.instance = instance
                self.fitness_func = fitness_func_

            def _evaluate(self, x, out, *args, **kwargs):
                out = self.fitness_func(self.instance, x, out)

                assert "solution" not in out, "Do not use `solution` key, it is pymoo reserved keyword"

                return out

        problem = VRPTW(instance, self.fitness_func)
        res = minimize(problem, self.algorithm, self.termination,
                       verbose=True, seed=self.seed,
                       callback=self.callback)

        print("Optimal solution:")
        print(res.X)
        print("Optimal value:")
        print(res.F)

        if update_history:
            X = np.floor(res.X).astype(int)
            d = {}
            problem._evaluate(X, d)

            start_times = d['start_times']
            fitness_value = max(start_times[i] + instance.durations[i] for i in range(len(instance.durations)))
            export = {"tasks_schedule": [
                {"start": start_times[i], "end": start_times[i] + instance.durations[i], "name": f"Task_{i}"} for i in
                range(instance.no_jobs)]}

            fitness_value = int(fitness_value)  # F - modified makespan (< 1)
            solution_info = f"start_times: {start_times}"
            solution_progress = res.algorithm.callback.data['progress']
            self.add_run_to_history(instance, fitness_value, solution_info, solution_progress,
                                    exec_time=round(res.exec_time, 2))

        # if res.F is not None:
        #     X = np.floor(res.X).astype(int)
        #     fitness_value = res.F[0]
        #     print('Objective value:', fitness_value)

        #     d = {}
        #     problem._evaluate(X, d)
        #     start_times = d['start_times']
        #     export = {"tasks_schedule": [{"start": start_times[i], "end": start_times[i] +
        #                                   instance.durations[i], "name": f"Task_{i}"} for i in range(instance.no_jobs)]}

        # if res.F is not None:
        #     return fitness_value, start_times, res
        # else:
        return None, None, res


if __name__ == "__main__":
    path = "..\\..\\..\\data\\VRPTW\\solomon_25\\C101.json"

    algorithm = PSO()
    instance = load_instance(path)
    solver = VRPTWSolver(algorithm=algorithm, fitness_func=fitness_func, termination=('n_gen', 100), seed=1, solver_name="GA")

    res = solver.solve(instance, validate=True, visualize=True)


# if __name__ == "__main__":
#     class VRPTW(ElementwiseProblem):
#         """pymoo wrapper class
#         """
#
#         def __init__(self, instance):
#             # print("number of customers", instance.nb_customers)
#             super().__init__(n_var=instance.nb_customers, n_obj=1,
#                              n_ieq_constr=4, n_eq_constr=4, xl=0, xu=100)
#             self.instance = instance
#
#         def _evaluate(self, x, out, *args, **kwargs):
#             out = fitness_func(self.instance, x, out)
#
#             assert "solution" not in out, "Do not use `solution` key, it is pymoo reserved keyword"
#
#             return out
#
#
#     from pymoo.algorithms.soo.nonconvex.ga import GA
#
#     path = "..\\..\\..\\data\\VRPTW\\solomon_25\\C101.json"
#     instance = load_instance(path)
#     algorithm = GA(
#         pop_size=100,
#         eliminate_duplicates=True)
#     problem = VRPTW(instance)
#     termination = ('n_gen', 100)
#     res = minimize(problem, algorithm, termination,
#                    verbose=True)
#
#     print("Optimal solution:")
#     print(res.X)
#     print("Optimal value:")
#     print(res.F)
