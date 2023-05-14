from docplex.cp.model import CpoModel
from src.optimization_problem import OptimizationProblem


class JobShop(OptimizationProblem):
    def __init__(self, benchmark_name, instance_name, format_, data, solution, run_history) -> None:
        super().__init__(benchmark_name, instance_name, format_, data, solution, run_history)

        self.no_jobs = self._data["no_jobs"]
        self.no_machines = self._data["no_machines"]
        self.durations = self._data["durations"]
        self.machines = self._data["machines"]

    def solve(self):
        model = CpoModel()

        x = [[model.interval_var(size=self.durations[job][order_index]) for order_index in range(self.no_machines)] for job in range(self.no_jobs)]
        model.add(
            [ model.minimize( model.max( model.end_of(x[i][self.no_machines-1]) for i in range(self.no_jobs) ) ) ] +
            [ model.no_overlap( x[job][machine_index] for job in range(self.no_jobs) for machine_index in range(self.no_machines) if self.machines[job][machine_index] == k ) for k in range(self.no_machines) ] +
            [ model.end_before_start( x[i][j-1], x[i][j] )  for i in range(self.no_jobs) for j in range(self.no_machines) if 0<j ]
        )

        print("Using 10 second time limit")
        sol = model.solve(TimeLimit=10, verbose=0, log_output=None)

        if sol:
            print("Project completion time:", sol.get_objective_value())
        else:
            print("No solution found.")

        return sol, x
