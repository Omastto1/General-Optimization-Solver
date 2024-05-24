import re
import multiprocessing
import functools
import time

from abc import ABC, abstractmethod

from docplex.cp.model import CpoParameters
from docplex.cp.solution import CpoSequenceVarSolution
from docplex.cp.expression import compare_expressions

from pymoo.core.callback import Callback

from src.utils import convert_time_to_seconds
from src.common.optimization_problem import Benchmark

from pydantic import BaseModel

SOLVER_DEFAULT_NAME = "Unknown solver - check whether solver_name is specified for solver class"


# TODO: Add solver path (to better identify solver if 2 solvers have the same name) and class name to output
class Solver(ABC):
    solver_name = SOLVER_DEFAULT_NAME

    def __init__(self):
        if self.solver_name == SOLVER_DEFAULT_NAME:
            print("\nWarning: solver_name not specified for solver\n")

    def solve(self, instance_or_benchmark, validate=False, visualize=False, force_execution=False, force_dump=None,
              hybrid_CP_solver=None, output=None):
        # in case of hybrid solver (GA + CP) do not save GA result into history
        update_history = False if hybrid_CP_solver is not None else True

        if isinstance(instance_or_benchmark, Benchmark):
            for instance_name, instance in instance_or_benchmark._instances.items():
                print(f"Solving instance {instance_name}...")
                _, solution, _ = self._solve(instance, validate=validate, visualize=visualize,
                                             force_execution=force_execution, update_history=update_history)

                # print(solution)

                if hybrid_CP_solver is not None:
                    _, solution, _ = hybrid_CP_solver._solve(instance, validate=validate, visualize=visualize,
                                                             force_execution=force_execution, initial_solution=solution)

            if force_dump is None:
                print("Force Dump not set, defaulting to saving the instances")
                force_dump = True

            if force_dump:
                instance_or_benchmark.dump(dir_path=output)
            elif output is not None:
                print("Output specified but force_dump is set to False, instances will not be saved")
        else:
            fitness_value, solution, res = self._solve(instance_or_benchmark, validate=validate, visualize=visualize,
                                                       force_execution=force_execution, update_history=update_history)

            if hybrid_CP_solver is not None:
                fitness_value, solution, res = hybrid_CP_solver._solve(instance_or_benchmark, validate=validate,
                                                                       visualize=visualize,
                                                                       force_execution=force_execution,
                                                                       initial_solution=solution)

            if force_dump is None:
                print("Force Dump not set, defaulting to saving the instances")
                force_dump = True

            if force_dump:
                instance_or_benchmark.dump(dir_path=output)
            elif output is not None:
                print("Output specified but force_dump is set to False, instances will not be saved")

            return fitness_value, solution, res

    @abstractmethod
    def _solve(self, instance, validate=False, visualize=False, force_execution=False, update_history=True):
        """Abstract solve method for solver."""
        pass


CP_SOLVER_DEFAULT_NAME = "CP solver without name specified"


class CPSolver(Solver):
    solver_name = CP_SOLVER_DEFAULT_NAME
    solver_type = "CP"

    def __init__(self, TimeLimit=60, no_workers=0):
        super().__init__()

        if self.solver_name == "CP_SOLVER_DEFAULT_NAME":
            # TODO: ADD SOLVER PARAMS TO NAME
            print("\nWarning: solver_name not specified for CP solver\n")

        self.solved = False
        # self.TimeLimit = TimeLimit
        self.params = CpoParameters()
        # params.SearchType = 'Restart'
        # self.params.LogPeriod = 100000
        self.params.LogVerbosity = 'Terse'
        self.params.TimeLimit = TimeLimit

        if no_workers > 0:
            self.params.Workers = no_workers

        print(
            f"Time limit set to {TimeLimit} seconds" if TimeLimit is not None else "Time limit not restricted")

    def _wrap_solve(self, instance, validate=False, visualize=False, force_execution=False):
        if not force_execution and len(instance._run_history) > 0:
            if instance.skip_on_optimal_solution():
                return None, None

        solution = self._solve(instance, validate, visualize)

        info = self.retrieve_solution_info(instance, solution)

        if solution:
            if validate:
                try:
                    print("Validating solution...")
                    instance.validate(solution, job_operations)
                    instance.validate(solution, job_operations)
                    print("Solution is valid.")
                except AssertionError as e:
                    print("Solution is invalid.")
                    print(e)
                    return None, None, None

            if visualize:
                instance.visualize(solution, job_operations, machine_operations)

            print("Project completion time:", solution.get_objective_values()[0])
        else:
            print("No solution found.")

        # print solution
        if solution.get_solve_status() == 'Optimal':
            print("Optimal solution found")
        elif solution.get_solve_status() == 'Feasible':
            print("Feasible solution found")
        else:
            print("Unknown solution status")
            print(solution.get_solve_status())

        obj_value = solution.get_objective_values()[0]
        print('Objective value:', obj_value)
        instance.compare_to_reference(obj_value)

        Solution = namedtuple("Solution", ['job_operations', 'machine_operations'])
        variables = Solution(job_operations, machine_operations)

        instance.update_run_history(sol, variables, "CP", self.params)

        return objective_value, info, cp_solution

    @abstractmethod
    def _solve(self, instance, validate, visualize, force_execution):
        """Abstract solve method for CP solver."""
        pass

    def _extract_solution_progress(self, log):
        #  Changed the pattern to match with my VRP, if it breaks something else, merge them or make a switch
        # r"\*\s+(\d+)\s+(?:\d+)\s+(\d+\.\d+s)"
        pattern = r"\*\s+(\d+\.\d+)\s+\w+\s+(\d+\.\d+s)"

        # Find all matches of numbers and times in the log using the regex pattern
        matches = re.findall(pattern, log, re.MULTILINE)

        # Convert minutes and hours into seconds and store the results
        result = [[float(match[0]), match[1]] for match in matches]
        solution_progress = convert_time_to_seconds(result)

        return solution_progress

    def parse_cp_solution_info(self, sol):
        """docplex.cp.solution.CpoSolveResult.write
        """
        info_dict = {}

        # Print model attributes
        sinfos = sol.get_solver_infos()
        info_dict["Model constraints"] = sinfos.get_number_of_constraints()
        info_dict["variables"] = {"integer": sinfos.get_number_of_integer_vars(),
                                  "interval": sinfos.get_number_of_interval_vars(),
                                  "sequence": sinfos.get_number_of_sequence_vars()}

        # Print search/solve status
        s = sol.get_search_status()
        if s:
            info_dict["Solve status"] = str(sol.get_solve_status())
            info_dict["Search status"] = str(s)
            s = sol.get_stop_cause()
            if s:
                info_dict["Search status stop cause"] = str(s)
        else:
            # Old fashion
            info_dict["Solve status"] = str(sol.get_solve_status())
            info_dict["Fail status"] = str(sol.get_fail_status())
        # Print solve time
        info_dict["Solve time"] = str(round(sol.get_solve_time(), 2)) + " sec"

        info_dict = self.parse_cp_model_solution_info(sol.solution, info_dict)

        return info_dict

    def parse_cp_model_solution_info(self, model_sol, info_dict):
        """docplex.cp.solution.CpoModelSolution.write
        """
        # Print objective value, bounds and gaps
        ovals = model_sol.get_objective_values()
        if ovals:
            info_dict["Objective values"] = ovals
        bvals = model_sol.get_objective_bounds()
        if bvals:
            info_dict["Bounds"] = bvals
        gvals = model_sol.get_objective_gaps()
        if gvals:
            info_dict["Gaps"] = gvals

        # Print all KPIs in declaration order
        kpis = model_sol.get_kpis()
        if kpis:
            info_dict["KPIs"] = {}
            for k in kpis.keys():
                info_dict["KPIs"][k] = kpis[k]

        # Print all variables in natural name order
        allvars = model_sol.get_all_var_solutions()
        if allvars:
            info_dict["Variables"] = {}
            lvars = [v for v in allvars if v.get_name()]
            lvars = sorted(lvars, key=functools.cmp_to_key(lambda v1, v2: compare_expressions(v1.expr, v2.expr)))
            for v in lvars:
                vval = v.get_value()
                if isinstance(v, CpoSequenceVarSolution):
                    vval = [iv.get_name() for iv in vval]
                info_dict["Variables"][v.get_name()] = vval
            nbanonym = len(allvars) - len(lvars)

            # TODO: IF VARIABLE IS NOT NAMED IN MODEL THE RESULT IS NOT EXPORTED
            if nbanonym > 0:
                info_dict["Variables"]["Anonymous variables"] = nbanonym

        return info_dict

    def add_run_to_history(self, instance, sol):
        """_summary_

        Args:
            instance (_type_): _description_
            sol (_type_): _description_
        """
        solution_progress = self._extract_solution_progress(sol.solver_log)

        if sol:
            objective_value = sol.get_objective_values()[0]
            # solution_info = sol.write_in_string()
            solution_info = self.parse_cp_solution_info(sol)
            solve_status = sol.get_solve_status()
            solve_time = sol.get_solve_time(),
            solution_progress = solution_progress
        else:
            objective_value = -1
            solution_info = {}
            solve_status = "No solution found"
            solve_time = self.params.TimeLimit
            solution_progress = []

        solver_name = self.solver_name
        solver_type = self.solver_type
        solver_config = {
            "TimeLimit": self.params.TimeLimit,
            "NoWorkers": sol.solver_infos['EffectiveWorkers'],
            "NoCores": multiprocessing.cpu_count(),
            "SolverVersion": sol.process_infos['SolverVersion']
        }

        instance.update_run_history(solver_name, solver_type, objective_value, solution_info,
                                    solve_status, solve_time, solver_config, solution_progress)


def serialize_class_instance(obj):
    """
    Serialize a class instance (including nested class instances) into a dictionary.
    """
    if not hasattr(obj, "__dict__"):
        return obj
    result = {}
    for key, val in obj.__dict__.items():
        # if isinstance(val, list):
        #     # Handle lists of items
        #     element = [serialize_class_instance(item) for item in val]
        #     result[key] = element
        if hasattr(val, "__dict__"):
            # Recursively serialize class instances
            result[key] = serialize_class_instance(val)
        else:
            result[key] = val
    return result


class HistoryCallback(Callback):

    def __init__(self, algorithm) -> None:
        super().__init__()
        self.data["progress"] = []
        self.algorithm_type = algorithm.__class__.__name__
        algorithm.start_time2 = time.perf_counter()

    def notify(self, algorithm):
        f_min = algorithm.pop.get("F").min()
        last_f_min = self.data["progress"][-1][0] if len(self.data["progress"]) > 0 else float('inf')

        if f_min < last_f_min:
            exec_time = round(time.perf_counter() - algorithm.start_time2, 2)

            if self.algorithm_type == "GA":
                no_individuals = algorithm.pop_size * algorithm.n_gen
            elif self.algorithm_type == "BRKGA":
                no_individuals = algorithm.n_elites + (algorithm.n_mutants + algorithm.n_offsprings) * algorithm.n_gen
            else:
                no_individuals = -1

            new_timestamp = (f_min, exec_time, no_individuals)

            self.data["progress"].append(new_timestamp)


GA_SOLVER_DEFAULT_NAME = "GA solver without name specified"


class GASolver(Solver):
    solver_name = GA_SOLVER_DEFAULT_NAME
    solver_type = "GA"

    def __init__(self, algorithm, fitness_func, termination, seed=None, solver_name=None):
        super().__init__()

        if solver_name is not None:
            self.solver_name = solver_name
        else:
            print("Warning: solver_name not specified for GA solver")

        self.algorithm = algorithm
        self.fitness_func = fitness_func
        self.termination = termination
        self.seed = seed
        self.callback = Callback()

    @abstractmethod
    def _solve(self, instance, validate, visualize, force_execution):
        """Abstract solve method for GP solver."""
        pass

    def add_run_to_history(self, instance, objective_value, solution_info, solution_progress, exec_time=-1,
                           is_valid=True):
        solve_time = exec_time

        if is_valid and objective_value >= 0:
            solve_status = "Feasible"
        elif not is_valid:
            solve_status = "Infeasible"
        else:
            solve_status = "No solution found"

        solver_name = self.solver_name
        solver_type = self.solver_type

        # TODO: FINISH - replace algorithm (function execution result) and fitness function (function definition)

        algorithm = serialize_class_instance(self.algorithm)
        solver_config = {
            "seed": self.seed,
            "algorithm": algorithm,
            "fitness_func": self.fitness_func,
            "termination": self.termination
        }

        instance.update_run_history(solver_name, solver_type, objective_value, solution_info,
                                    solve_status, solve_time, solver_config, solution_progress)


class ORtoolsSolver(Solver):
    solver_name = CP_SOLVER_DEFAULT_NAME
    solver_type = "CP"

    def __init__(self, TimeLimit=60, no_workers=0):
        super().__init__()

        if self.solver_name == "CP_SOLVER_DEFAULT_NAME":
            # TODO: ADD SOLVER PARAMS TO NAME
            print("\nWarning: solver_name not specified for CP solver\n")

        self.solved = False
        # self.TimeLimit = TimeLimit
        self.params = CpoParameters()
        # params.SearchType = 'Restart'
        # self.params.LogPeriod = 100000
        self.params.LogVerbosity = 'Terse'
        self.params.TimeLimit = TimeLimit

        if no_workers > 0:
            self.params.Workers = no_workers

        print(
            f"Time limit set to {TimeLimit} seconds" if TimeLimit is not None else "Time limit not restricted")

    @abstractmethod
    def _solve(self, instance, validate, visualize, force_execution):
        """Abstract solve method for GP solver."""
        pass

    def add_run_to_history(self, instance, sol, result, history, time):
        """_summary_

        Args:
            instance (_type_): _description_
            sol (_type_): _description_
        """
        solution_progress = history

        # TODO: Fix history objective values
        # diff = result['total_distance'] - float(str(solution_progress[-1][0])[2:])
        # solution_progress = [[round(float(str(i[0])[2:]) + diff, 2), i[1]] for i in solution_progress]

        if sol:
            objective_value = result
            # solution_info = sol.write_in_string()
            solution_info = []
            solve_status = "Feasible"
            solve_time = time
            solution_progress = solution_progress
        else:
            objective_value = -1
            solution_info = {}
            solve_status = "No solution found"
            solve_time = time
            solution_progress = []

        solver_name = self.solver_name
        solver_type = self.solver_type
        solver_config = {
            "TimeLimit": self.params.TimeLimit,
            "NoCores": multiprocessing.cpu_count(),
        }

        instance.update_run_history(solver_name, solver_type, objective_value, solution_info,
                                    solve_status, solve_time, solver_config, solution_progress)
