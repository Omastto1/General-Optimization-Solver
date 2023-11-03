import re
import multiprocessing

from docplex.cp.model import CpoParameters
from abc import ABC, abstractmethod

from src.utils import convert_time_to_seconds
from src.common.optimization_problem import Benchmark


class Solver(ABC):
    def solve(self, instance_or_benchmark, **kwargs):
        if isinstance(instance_or_benchmark, Benchmark):
            for instance_name, instance in instance_or_benchmark._instances.items():
                self._solve(instance, **kwargs)
            # return self.solve_benchmark(instance_or_benchmark)
        else:
            return self._solve(instance_or_benchmark, **kwargs)
        pass


class CPSolver(Solver):
    def __init__(self, TimeLimit=60, no_workers=0):
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
    def _solve(self):
        """Abstract solve method for CP solver."""
        pass

    def _extract_solution_progress(self, log):
        pattern = r"\*\s+(\d+)\s+(?:\d+)\s+(\d+\.\d+s)"

        # Find all matches of numbers and times in the log using the regex pattern
        matches = re.findall(pattern, log, re.MULTILINE)

        # Convert minutes and hours into seconds and store the results
        result = [[int(match[0]), match[1]] for match in matches]
        solution_progress = convert_time_to_seconds(result)

        return solution_progress

    def add_run_to_history(self, instance, sol):
        solution_progress = self._extract_solution_progress(sol.solver_log)

        if sol:
            objective_value = sol.get_objective_values()[0]
            solution_info = sol.write_in_string()
            solve_status = sol.get_solve_status()
            solve_time = sol.get_solve_time(),
            solution_progress = solution_progress
        else:
            objective_value = -1
            solution_info = ""
            solve_status = "No solution found"
            solve_time = self.params.TimeLimit
            solution_progress = []

        solver_config = {
            "TimeLimit": self.params.TimeLimit,
            "NoWorkers": sol.solver_infos['EffectiveWorkers'],
            "NoCores": multiprocessing.cpu_count(),
            "SolverVersion": sol.process_infos['SolverVersion']
        }

        instance.update_run_history("CP", objective_value, solution_info,
                                    solve_status, solve_time, solver_config, solution_progress)


class GASolver(Solver):
    def __init__(self, algorithm, fitness_func, termination, seed=None):
        self.algorithm = algorithm
        self.fitness_func = fitness_func
        self.termination = termination
        self.seed = seed

    @abstractmethod
    def _solve(self):
        """Abstract solve method for GP solver."""
        pass

    def add_run_to_history(self, instance, objective_value, solution_info, is_valid=True):
        # TODO
        solution_progress = []
        solve_time = ""
        
        if is_valid and objective_value >= 0:
            solve_status = "Feasible" 
        elif not is_valid:
            solve_status = "Infeasible"
        else:
            solve_status = "No solution found"

        solver_config = {
            "seed": self.seed
        }

        instance.update_run_history("GA", objective_value, solution_info,
                                    solve_status, solve_time, solver_config, solution_progress)