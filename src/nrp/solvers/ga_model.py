from copy import deepcopy

import numpy as np
from pymoo.algorithms.soo.nonconvex.pso import PSO

from pymoo.core.problem import ElementwiseProblem, Problem
from pymoo.optimize import minimize

# from src.rcpsp.solvers.solver_cp import RCPSPCPSolver
# from src.rcpsp.solvers.solver_ga import RCPSPGASolver
from src.general_optimization_solver import load_raw_instance, load_instance, load_raw_benchmark
from src.common.solver import GASolver, HistoryCallback
from src.nrp.problem import validate_nrp, visualize_nrp
from src.vrp.problem import *

from queue import PriorityQueue

def decode_chromosome(instance, chromosome):
    pass


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
    routes, dist = decode_chromosome(instance, chromosome)
    # res1 = decode_chromosome_fast(instance, chromosome)
    # res2 = decode_chromosome_second(instance, chromosome)

    # routes, dist = min([res1, res2], key=lambda z: z[1])

    # print(f"DISTANCE: {dist}")

    out["F"] = dist
    # out["routes"] = {0: routes}     # For some reason, pymoo requires routes to be in a structure, string and dicts work

    # print(out)

    return out


class NRPSolver(GASolver):
    """GA SOLVER WRAPPER CLASS
    """

    def _solve(self, instance, validate=False, visualize=False, force_execution=False, update_history=True):
        class NRP(ElementwiseProblem):
            """pymoo wrapper class
            """

            def __init__(self, instance, fitness_func_):
                # print("number of customers", instance.nb_customers)
                num_vars = len(instance.staff) * instance.horizon
                super().__init__(n_var=num_vars, n_obj=1, xl=0, xu=len(instance.shifts)-1)
                self.instance = instance
                # self.fitness_func = fitness_func_

            def _evaluate(self, x, out, *args, **kwargs):
                # Decode chromosome into schedule
                schedule = self.decode_chromosome(x)

                # Calculate penalty and objective function value
                penalty = self.calculate_penalty(schedule)

                out["F"] = penalty
                # out = self.fitness_func(self.instance, x, out)

                assert "solution" not in out, "Do not use `solution` key, it is pymoo reserved keyword"

                return out

            def decode_chromosome(self, x):
                schedule = {}
                idx = 0
                for nurse in self.instance.staff.keys():
                    schedule[nurse] = []
                    for day in range(self.instance.horizon):
                        shift = x[idx]
                        schedule[nurse].append(shift)
                        idx += 1
                return schedule

            def calculate_penalty(self, schedule):
                # Implement the penalty calculations based on the CP model constraints and objectives
                # This should mirror the penalty calculation in the CP model

                # Initialize penalty
                penalty = 0
                print(schedule)

                # Hard and soft constraints with associated penalties
                # Example: Employees cannot be assigned more than one shift on a day
                for nurse in schedule:
                    for day in range(self.instance.horizon):
                        shifts = schedule[nurse][day]
                        if sum(shifts) > 1:
                            penalty += 1000  # Penalty for violating hard constraints

                # Add penalties for other constraints similarly

                return penalty

        self.callback = HistoryCallback(self.algorithm)
        problem = NRP(instance, self.fitness_func)
        res = minimize(problem, self.algorithm, self.termination,
                       verbose=False, seed=self.seed,
                       callback=self.callback)

        if res.F is None:
            print("No solution found")
            return None, None, res

        print(res)

        # chromosome = sorted(range(1, len(res.X) + 1), key=lambda i: res.X[i - 1])

        # routes, dist = decode_chromosome_fast(instance, chromosome)

        # res1 = decode_chromosome_fast(instance, chromosome)
        # res2 = decode_chromosome_second(instance, chromosome)
        #
        # routes, dist = min([res1, res2], key=lambda z: z[1])

        # fitness_value = dist / 10

        # print(f"Fitness value: {fitness_value}"
        #       f"\nRoutes: {routes}")
        penalty = res.F
        roster = problem.decode_chromosome(res.X)

        export = {'roster': roster, 'penalty': penalty}

        if validate:
            try:
                print("Validating solution...")
                is_valid = validate_nrp(roster, instance, penalty)
                if is_valid:
                    print("Solution is valid.")
                else:
                    print("Solution is invalid.")
            except AssertionError as e:
                print("Solution is invalid.")
                print(e)
                return None, export, res

        if visualize:
            visualize_nrp(roster, instance, penalty)

        if update_history:
            fitness_value = penalty
            solution_info = export
            solution_progress = deepcopy(res.algorithm.callback.data['progress'])
            self.add_run_to_history(instance, fitness_value, solution_info, solution_progress,
                                    exec_time=round(res.exec_time, 2))

        return fitness_value, export, res


if __name__ == "__main__":
    path = "..\\..\\..\\data\\NRP\\NRC\\Instance1.json"

    algorithm = PSO()
    instance = load_instance(path)
    solver = NRPSolver(algorithm=algorithm, fitness_func=fitness_func, termination=('n_gen', 1000), solver_name="GA")

    res = solver.solve(instance, validate=True, visualize=True)

    print(res)

    print(instance)

    instance.dump(verbose=True, dir_path="..\\..\\..\\data\\NRP")
