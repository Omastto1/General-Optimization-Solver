import datetime
import json

from pathlib import Path
from typing import Optional, List


class Benchmark:
    def __init__(self, name, instances) -> None:
        self._name: str = name
        self._instances: dict = instances
        # self._format: str = format

    def __str__(self):
        return "Benchmark"

    def __repr__(self):
        return "Benchmark"

    def dump(self, dir_path: Optional[str] = None):
        """Dumps all instances to their respective paths (defined by benchmark name + instance name)
        Optionally, a directory path can be specified to dump all instances to specific directory (other than the default benchmark directory)

        Args:
            dir_path (Optional[str], optional): Alternative directory where instance should be saved. Defaults to None.
        """
        print("Dumping instances to their respective paths")
        for _, instance in self._instances.items():
            instance.dump(dir_path=dir_path)

    def generate_solver_comparison_markdown_table(self, instances_subset=None, solvers_subset=None) -> str:
        """Generates a markdown table comparing the solvers on the instances

        Example: (Star marks optimal objective value)
        | Instance | BRKGA_forward | CP Default | 
        | -- | -- | -- | 
        | j3010_1 | 43.0 | 42* | 
        | j3010_10 | 42.0 | 41* | 
        | j3010_2 | 58.0 | 56* | 

        Args:
            instances_subset (Optional[list[str]], optional): Names of instances that should be included in the output. Defaults to None.
            solvers_subset (Optional[list[str]], optional): Names of solvers that should be included in the output. Defaults to None.

        Returns:
            str: markdown table as described above
        """
        if instances_subset is None:
            instances_subset = self._instances.keys()
        if solvers_subset is None:
            temp_solvers_subset = set()

        table_data = {instance_name: {} for instance_name in instances_subset}

        for instance_name, instance in self._instances.items():
            if instance_name in instances_subset:
                for instance_run in instance._run_history:
                    if solvers_subset is None or instance_run["solver_name"] in solvers_subset:
                        print(instance_run["solver_name"],
                              instance_run["solve_time"])
                        table_data[instance_name][instance_run["solver_name"]] = {
                            "objective_value": instance_run["solution_value"]}

                        if instance_run['solver_type'] != "CP":
                            optimum_known = instance._solution.get('optimum') is not None
                            
                            if optimum_known:
                                solution_status = "Optimal" if instance._solution.get(
                                    'optimum') == instance_run['solution_value'] else ""
                            else:
                                lower_bound =instance._solution.get('bounds', {}).get("lower")
                                if instance_run['solution_value'] == lower_bound:
                                    solution_status = "Lower bound"
                                elif instance_run['solution_value'] < lower_bound:
                                    solution_status = "Better than lower bound"
                                else:
                                    solution_status = ""
                            
                        else:
                            solution_status = instance_run.get(
                                "solve_status", "Unknown")

                        table_data[instance_name][instance_run["solver_name"]
                                                  ]["solution_status"] = solution_status

                        if solvers_subset is None:
                            temp_solvers_subset.add(
                                instance_run["solver_name"])

        if solvers_subset is None:
            solvers_subset = list(temp_solvers_subset)

        # empty start and end to force " | " to start and end the line
        column_headers = [""] + ["Instance"] + solvers_subset + [""]
        table_markdown = " | ".join(column_headers).strip() + "\n"

        column_header_body_delimiter = [
            ""] + [" -- "] * (len(column_headers) - 2) + [""]
        table_markdown += " | ".join(
            column_header_body_delimiter).strip() + "\n"

        sorted_table_data = sorted(table_data.items(), key=lambda x: x[0])
        for instance_name, instance_data in sorted_table_data:
            table_markdown += f"| {instance_name} | "
            for method in solvers_subset:
                if method in instance_data:
                    # TODO: DEPENDS ON CP OUTPUT, NOT WORKING FOR GA
                    is_optimal_objective = instance_data[method]["solution_status"] == "Optimal"
                    is_lower_bound_objective = instance_data[method]["solution_status"] == "Lower bound"
                    is_better_than_lower_bound_objective = instance_data[method]["solution_status"] == "Better than lower bound"

                    table_markdown += f"{instance_data[method]['objective_value']}"
                    if is_optimal_objective:
                        table_markdown += "*"
                    if is_lower_bound_objective:
                        table_markdown += "**"
                    if is_better_than_lower_bound_objective:
                        table_markdown += "***"

                    table_markdown += " | "
                else:
                    table_markdown += f"{instance_data.get(method, 'N/A')} | "
            table_markdown += "\n"

        return table_markdown

    def generate_solver_comparison_percent_deviation_markdown_table(self, instances_subset: Optional[List[str]] = None, solvers_subset: Optional[List[str]] = None):
        """Generates a markdown table that lists average solver deviation from benchmark objective values

        Example:
        | solver | deviation (%) |
        | -- | -- |
        | CP Default | 0.0 | 
        | naive GA backward | 6.8 | 
        | naive GA forward | 4.5 | 

        Args:
            instances_subset (Optional[List[str]], optional): Names of instances that should be included in the output. Defaults to None.
            solvers_subset (Optional[List[str]], optional): Names of solvers that should be included in the output. Defaults to None.

        Returns:
            _type_: _description_
        """
        if all(instance._solution == {} for instance in self._instances.values()):
            print("No solution found for instance")
            return

        if instances_subset is None:
            instances_subset = self._instances.keys()
        if solvers_subset is None:
            temp_solvers_subset = set()

        table_data = {}

        for instance_name, instance in self._instances.items():
            if instance_name in instances_subset:
                for instance_run in instance._run_history:
                    if solvers_subset is None or instance_run["solver_name"] in solvers_subset:
                        print(instance_run["solver_name"],
                              instance_run["solve_time"])
                        if instance_run["solver_name"] not in table_data:
                            table_data[instance_run["solver_name"]] = {}

                        if instance._solution.get('optimum') is not None:
                            table_data[instance_run["solver_name"]][instance_name] = {"deviation": 100 * instance_run["solution_value"] / instance._solution.get(
                                'optimum') - 100,
                                
                                "time": instance_run['solve_time'][0] if isinstance(instance_run['solve_time'], list) else instance_run['solve_time']}
                        elif instance._solution.get('bounds', {}).get('lower') is not None:
                            table_data[instance_run["solver_name"]][instance_name] = {"deviation": 100 * instance_run["solution_value"] / instance._solution.get('bounds', {}).get('lower') - 100,
                                "time":  instance_run['solve_time'][0] if isinstance(instance_run['solve_time'], list) else instance_run['solve_time']}
                        

                        if solvers_subset is None:
                            temp_solvers_subset.add(
                                instance_run["solver_name"])

        if solvers_subset is None:
            solvers_subset = list(temp_solvers_subset)

        # empty start and end to force " | " to start and end the line
        column_headers = [""] + ["solver"] + ["deviation (%)"] + ["time (s)"] + [""]
        table_markdown = " | ".join(column_headers).strip() + "\n"

        column_header_body_delimiter = [
            ""] + [" -- "] * (len(column_headers) - 2) + [""]
        table_markdown += " | ".join(
            column_header_body_delimiter).strip() + "\n"

        for solver_name, solver_data in table_data.items():
            table_markdown += f"| {solver_name} | "
            avg_deviation = round(
                sum(solver['deviation'] for solver in solver_data.values()) / len(solver_data.values()), 1)
            avg_time = round(
                sum(solver['time'] for solver in solver_data.values()) / len(solver_data.values()), 1)

            table_markdown += f"{avg_deviation} | {avg_time} | \n"

        return table_markdown


class OptimizationProblem:
    def __init__(self, benchmark_name, instance_name, _instance_kind, data, solution, run_history) -> None:
        assert isinstance(run_history, list)
        assert isinstance(solution, dict)

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

    def dump(self, verbose: bool = False, dir_path: Optional[str] = None) -> None:
        """Dumps instance to its respective path (defined by benchmark name + instance name)
        Optionally, a directory path can be specified to dump the instances to specific directory (other than the default benchmark directory)

        Args:
            dir_path (Optional[str], optional): Alternative directory where instance should be saved. Defaults to None.
        """
        instance_dict = {
            "benchmark_name": self._benchmark_name,
            "instance_name": self._instance_name,
            "instance_kind": self._instance_kind,

            "data": self._data,
            "reference_solution": self._solution,
            "run_history": self._run_history,
        }

        if dir_path is not None:
            if not dir_path.endswith("/"):
                dir_path += "/"
            benchmark_directory = dir_path   
        else:
            benchmark_directory = f"data/{self._instance_kind}/{self._benchmark_name}/"

        Path(benchmark_directory).mkdir(parents=True, exist_ok=True)

        path = benchmark_directory + f"{self._instance_name}.json"

        if verbose:
            print("dumping to", path)

        with open(path, "w+", encoding='utf-8') as f:
            json.dump(instance_dict, f, indent=4, default=str)

    def compare_to_reference(self, obj_value):
        # TODO: WHAT HAPPENS IF SELF SOLUTION IS NOT FEASIBLE?
        if self._solution.get("optimum", None) is not None:
            if obj_value == self._solution["optimum"]:
                print("Solution is optimal.")
            else:
                ratio = round(
                    (obj_value / self._solution["optimum"] - 1) * 100, 1)
                print(f"Solution is {ratio} % worse than the optimum.")
        elif self._solution.get("bounds", None) is not None:
            if obj_value >= self._solution["bounds"]["lower"]:
                ratio = round(
                    (obj_value / self._solution["bounds"]["lower"] - 1) * 100, 1)
                print(f"Solution is {ratio} % worse than the lower bound.")
            else:
                ratio = round(
                    (obj_value / self._solution["bounds"]["upper"] - 1) * 100, 1)
                print(f"Solution is {ratio} % worse than the upper bound.")
        else:
            print("There in no known reference solution in current data")

    def update_run_history(self, method, solver_type, objective_value, solution_info, solve_status, solve_time, solver_config, solution_progress):
        """Updates the run history of the instance with the given information

        Args:
            method (_type_): _description_
            solver_type (_type_): _description_
            objective_value (_type_): _description_
            solution_info (_type_): _description_
            solve_status (_type_): _description_
            solve_time (_type_): _description_
            solver_config (_type_): _description_
            solution_progress (_type_): _description_
        """
        timestamp_now = datetime.datetime.now()

        self._run_history.append({
            "timestamp": timestamp_now,
            "solver_type": solver_type,
            "solver_name": method,
            "solver_config": solver_config,
            "solve_status": solve_status,  # docplex specific so far
            "solve_time": solve_time,  # docplex specific so far
            "solution_value": objective_value,
            "solution_info": solution_info,  # docplex specific so far
            "solution_progress": solution_progress  # docplex specific so far
        })

    def reset_run_history(self):
        """Resets the run history of the instance, i.e. removes all entries from the run history
        """
        self._run_history = []

    def skip_on_optimal_solution(self) -> bool:
        """Checks if the instance has already been solved optimally

        Returns:
            bool: True if the instance has already been solved optimally, False otherwise
        """
        is_solved_optimally = self._run_history[-1]["solution_value"] == self._solution.get(
            "optimum", None)
        if is_solved_optimally:
            print("Instance already solved optimally.")
            print("Skipping...")
            return True

        return False
