from docplex.cp.model import CpoModel
from src.optimization_problem import OptimizationProblem
import docplex.cp.utils_visu as visu


class JobShop(OptimizationProblem):
    def __init__(self, benchmark_name, instance_name, data, solution, run_history) -> None:
        super().__init__(benchmark_name, instance_name, "JOBSHOP", data, solution, run_history)

        self.no_jobs = self._data["no_jobs"]
        self.no_machines = self._data["no_machines"]
        self.durations = self._data["durations"]
        self.machines = self._data["machines"]

    def validate(self, sol, jobs):
        for jobs_temp in jobs:
            for i in range(1, self.no_machines):
                assert sol.get_var_solution(jobs_temp[i-1]).get_end() <= sol.get_var_solution(
                    jobs_temp[i]).get_start(), f"Job {i-1} ends after job {i} starts."

        return True

    def visualize(self, sol, job_operations, machine_operations):
        if sol and visu.is_visu_enabled():
            print(1, sol.get_objective_value())
            visu.timeline('Solution for job-shop ' +
                          self._instance_name, 1, sol.get_objective_value())
            visu.panel('Jobs')
            for i in range(self.no_jobs):
                visu.sequence(name='J' + str(i),
                              intervals=[(sol.get_var_solution(job_operations[i][j]), self.machines[i][j], 'M' + str(self.machines[i][j])) for j in
                                         range(self.no_machines)])
            visu.panel('Machines')
            for k in range(self.no_machines):
                visu.sequence(name='M' + str(k),
                              intervals=[(sol.get_var_solution(machine_operations[k][i]), k, 'J' + str(i)) for i in range(self.no_jobs)])
            visu.show()
