import datetime
import json

from pathlib import Path
from typing import Optional



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

    def generate_solver_comparison_markdown(self, instances_subset=None, methods_subset=None):
        if instances_subset is None:
            instances_subset = self._instances.keys()
        if methods_subset is None:
            temp_methods_subset = set()
    
        table_data = {instance_name: {} for instance_name in instances_subset}

        for instance_name, instance in self._instances.items():
            if instance_name in instances_subset:
                for instance_run in instance._run_history:
                    if methods_subset is None or instance_run["method"] in methods_subset:
                        print(instance_run["method"], instance_run["solve_time"])
                        table_data[instance_name][instance_run["method"]] = instance_run["solution_value"]

                        if methods_subset is None:
                            temp_methods_subset.add(instance_run["method"])
                            
        if methods_subset is None:
            methods_subset = list(temp_methods_subset)

        column_headers = [""] + ["Instance"] + methods_subset + [""]  # empty start and end to force " | " to start and end the line
        table_markdown = " | ".join(column_headers).strip() + "\n"

        column_header_body_delimiter = [""] + [" -- "] * (len(column_headers) - 2) + [""]
        table_markdown += " | ".join(column_header_body_delimiter).strip() + "\n"

        for instance_name, instance_data in table_data.items():
            table_markdown += f"| {instance_name} | "
            for method in methods_subset:
                table_markdown += f"{instance_data.get(method, 'N/A')} | "
            table_markdown += "\n"

        return table_markdown


class OptimizationProblem:
    def __init__(self, benchmark_name, instance_name, _instance_kind, data, solution, run_history) -> None:
        self._benchmark_name: str = benchmark_name
        self._instance_name: str = instance_name
        self._instance_kind: str = _instance_kind
        self._data: dict = data
        self._solution: Optional[dict] = solution
        self._run_history: Optional[list] = run_history

    def __str__(self):
        return "Optimization Problem"

    def __repr__(self):
        return "Optimization Problem"

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
        # TODO: WHAT HAPPENS IF SELF SOLUTION IS NOT FEASIBLE?
        if self._solution.get("optimum", None) is not None:
            if obj_value == self._solution["optimum"]:
                print("Solution is optimal.")
            else:
                ratio = round((obj_value / self._solution["optimum"] - 1) * 100, 1)
                print(f"Solution is {ratio} % worse than the optimum.")
        elif self._solution.get("bounds", None) is not None:
            if obj_value >= self._solution["bounds"]["lower"]:
                ratio = round((obj_value / self._solution["bounds"]["lower"] - 1) * 100, 1)
                print(f"Solution is {ratio} % worse than the lower bound.")
            else:
                ratio = round((obj_value / self._solution["bounds"]["upper"] - 1) * 100, 1)
                print(f"Solution is {ratio} % worse than the upper bound.")
        else:
            print("There in no known reference solution in current data")

    def update_run_history(self, method, objective_value, solution_info, solve_status, solve_time, solver_config, solution_progress):
        timestamp_now = datetime.datetime.now()
        
        self._run_history.append({
            "timestamp": timestamp_now,
            "method": method,
            "solution_value": objective_value,
            "solution_info": solution_info,  # docplex specific so far
            "solve_status": solve_status,  # docplex specific so far
            "solve_time": solve_time,  # docplex specific so far
            "solver_config": solver_config,
            "solution_progress": solution_progress  # docplex specific so far
        })
    
    def reset_run_history(self):
        self._run_history = []
    
    def skip_on_optimal_solution(self):
        is_solved_optimally = self._run_history[-1]["solution_value"] == self._solution.get("optimum", None)
        if is_solved_optimally:
            print("Instance already solved optimally.")
            print("Skipping...")
            return True
            
        return False