from typing import Optional


class Benchmark:
    def __init__(self, name, instances, format) -> None:
        self._name: str = name
        self._instances: dict = instances
        self._format: str = format

    def __str__(self):
        return "Benchmark"

    def __repr__(self):
        return "Benchmark"
    
    def solve(self):
        i = 1
        for instance_name, instance in self._instances.items():
            print("solving", instance_name)
            instance.solve()
            if i == 10:
                print("Ending after 10 iterations")
                break
            i += 1


class OptimizationProblem:
    def __init__(self, benchmark_name, instance_name, format_, data, solution, run_history) -> None:
        self._benchmark_name: str = benchmark_name
        self._instance_name: str = instance_name
        self._format: str = format_
        self._data: dict = data
        self._solution: Optional[dict] = solution
        self._run_history: Optional[dict] = run_history

    def __str__(self):
        return "Optimization Problem"

    def __repr__(self):
        return "Optimization Problem"

    def load(self, path):
        pass
