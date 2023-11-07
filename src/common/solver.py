import re
import multiprocessing
import functools

from docplex.cp.model import CpoParameters
from abc import ABC, abstractmethod

from src.utils import convert_time_to_seconds
from src.common.optimization_problem import Benchmark

from docplex.cp.solution import CpoSequenceVarSolution
from docplex.cp.expression import compare_expressions


class Solver(ABC):
    def solve(self, instance_or_benchmark, force_dump=None, **kwargs):
        if isinstance(instance_or_benchmark, Benchmark):
            for instance_name, instance in instance_or_benchmark._instances.items():
                self._solve(instance, **kwargs)
            # return self.solve_benchmark(instance_or_benchmark)

            if force_dump is None:
                print("Force Dump not set, defaulting to saving the instances")
                force_dump = True
            
            if force_dump:
                instance_or_benchmark.dump()
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
            if nbanonym > 0:
                info_dict["Variables"]["Anonymous variables"] = nbanonym

        return info_dict

    def add_run_to_history(self, instance, sol):
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