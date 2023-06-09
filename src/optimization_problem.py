import datetime
import json
import re
import multiprocessing

from pathlib import Path
from typing import Optional

from src.utils import convert_time_to_seconds


class Benchmark:
    def __init__(self, name, instances) -> None:
        self._name: str = name
        self._instances: dict = instances
        # self._format: str = format

    def __str__(self):
        return "Benchmark"

    def __repr__(self):
        return "Benchmark"
    
    def solve(self, solver, method="CP", force_dump=True):
        i = 1
        for instance_name, instance in self._instances.items():
            print("solving", instance_name)
            solver.solve(instance, method)
            # if i == 10:
            #     print("Ending after 10 iterations")
            #     break
            i += 1

        if force_dump:
            self.dump()

    def dump(self):
        print("Dumping instances to their respective paths")
        for instance_name, instance in self._instances.items():
            instance.dump_json()



class OptimizationProblem:
    def __init__(self, benchmark_name, instance_name, _instance_kind, data, solution, run_history) -> None:
        self._benchmark_name: str = benchmark_name
        self._instance_name: str = instance_name
        self._instance_kind: str = _instance_kind
        # self._format: str = format_
        self._data: dict = data
        self._solution: Optional[dict] = solution
        self._run_history: Optional[list] = run_history

    def __str__(self):
        return "Optimization Problem"

    def __repr__(self):
        return "Optimization Problem"

    def load(self, path):
        pass

    def dump(self, verbose=False):
        instance_dict = {
            "benchmark_name": self._benchmark_name,
            "instance_name": self._instance_name,
            "instance_kind": self._instance_kind,

            "data": self._data,
            "reference_solution": self._solution,
            "run_history": self._run_history,
        }
        benchmark_directory = f"data/{self._instance_kind}/{self._benchmark_name}/"
        Path(benchmark_directory).mkdir(parents=True, exist_ok=True)

        path = benchmark_directory + f"{self._instance_name}.json"

        if verbose:
            print("dumping to", path)

        with open(path, "w+") as f:
            json.dump(instance_dict, f, indent=4, default=str)

    def compare_to_reference(self, obj_value):
        if self._solution["optimum"] is not None:
            if obj_value == self._solution["optimum"]:
                print("Solution is optimal.")
            else:
                ratio = round((obj_value / self._solution["optimum"] - 1) * 100, 1)
                print(f"Solution is {ratio} % worse than the optimum.")
        elif self._solution["bounds"] is not None:
            if obj_value >= self._solution["bounds"]["lower"]:
                ratio = round((obj_value / self._solution["bounds"]["lower"] - 1) * 100, 1)
                print(f"Solution is {ratio} % worse than the lower bound.")
            else:
                ratio = round((obj_value / self._solution["bounds"]["upper"] - 1) * 100, 1)
                print(f"Solution is {ratio} % worse than the upper bound.")
        else:
            print("There in no known reference solution in current data")
 
    def update_run_history(self, sol, variables, method, solver_params):
        timestamp_now = datetime.datetime.now()\
        
        log = sol.solver_log

        # Define the regex pattern
        # pattern = r"\*\s+(\d+)\s+(\d+)\s+(\d+\.\d+s)"
        pattern = r"\*\s+(\d+)\s+(?:\d+)\s+(\d+\.\d+s)"

        # Find all matches of numbers and times in the log using the regex pattern
        matches = re.findall(pattern, log, re.MULTILINE)

        # Convert minutes and hours into seconds and store the results
        # result = [[int(match[0]), int(match[1]), match[2]] for match in matches]
        result = [[int(match[0]), match[1]] for match in matches]
        result = convert_time_to_seconds(result)

        if sol:
            self._run_history.append({
                "timestamp": timestamp_now,
                "method": method,
                "solution_value": sol.get_objective_values()[0],
                "solution_info": sol.write_in_string(),
                "solve_status": sol.get_solve_status(),
                "solve_time": sol.get_solve_time(),
                "solver_config": {
                    "TimeLimit": solver_params.TimeLimit,
                    "NoWorkers": sol.solver_infos['EffectiveWorkers'],
                    "NoCores": multiprocessing.cpu_count(),
                    "SolverVersion": sol.process_infos['SolverVersion']
                },
                "solution_progress": result
            })
        else:
            self._run_history.append({
                "timestamp": timestamp_now,
                "method": method,
                "solution_value": -1,
                "solve_status": "No solution found",
                "solve_time": solver_params.TimeLimit,
                "solver_config": {
                    "TimeLimit": solver_params.TimeLimit,
                    "NoWorkers": sol.solver_infos['EffectiveWorkers'],
                    "NoCores": multiprocessing.cpu_count(),
                    "SolverVersion": sol.process_infos['SolverVersion']
                },
                "solution_progress": []
            })
    
    def reset_run_history(self):
        self._run_history = []
    
    def skip_on_optimal_solution(self):
        is_solved_optimally = self._run_history[-1]["solution_value"] == self._solution["optimum"]
        is_solved_better_than_upper_bound = "upper_bound" in self._solution and self._run_history[-1]["solution_value"] <= self._solution["upper_bound"]
        if is_solved_optimally or is_solved_better_than_upper_bound:
            if self._run_history[-1]["solution_value"] == self._solution["optimum"]:
                print("Instance already solved optimally.")
                print("Skipping...")
                return True
            
        return False