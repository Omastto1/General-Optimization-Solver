from docplex.cp.model import CpoModel
from src.optimization_problem import OptimizationProblem
import docplex.cp.utils_visu as visu


class JobShop(OptimizationProblem):
    def __init__(self, benchmark_name, instance_name, format_, data, solution, run_history) -> None:
        super().__init__(benchmark_name, instance_name, format_, data, solution, run_history)

        self.no_jobs = self._data["no_jobs"]
        self.no_machines = self._data["no_machines"]
        self.durations = self._data["durations"]
        self.machines = self._data["machines"]

    def solve(self, validate=False, visualize=False):
        model = CpoModel()

        job_operations = [[model.interval_var(name=f"J_{job}_{order_index}", size=self.durations[job][order_index])
                           for order_index in range(self.no_machines)] for job in range(self.no_jobs)]
        model.add(
            [model.minimize(model.max(model.end_of(job_operations[i][self.no_machines-1]) for i in range(self.no_jobs)))] +
            [model.end_before_start(job_operations[i][j-1], job_operations[i][j])
             for i in range(self.no_jobs) for j in range(self.no_machines) if 0 < j]
        )

        # [ model.no_overlap( job_operations[job][machine_index] for job in range(self.no_jobs) for machine_index in range(self.no_machines) if self.machines[job][machine_index] == k ) for k in range(self.no_machines) ] +
        machine_operations = [[] for m in range(self.no_machines)]
        for j in range(self.no_jobs):
            for s in range(self.no_machines):
                machine_operations[self.machines[j]
                                   [s]].append(job_operations[j][s])
        for mops in machine_operations:
            model.add(model.no_overlap(mops))

        print("Using 10 second time limit")
        sol = model.solve(TimeLimit=10, verbose=0, log_output=None)

        if sol:
            if validate:
                try:
                    print("Validating solution...")
                    self.validate(sol, job_operations)
                    print("Solution is valid.")
                except AssertionError as e:
                    print("Solution is invalid.")
                    print(e)
                    return None, None, None

            if visualize:
                self.visualize(sol, job_operations, machine_operations)

            print("Project completion time:", sol.get_objective_value())
        else:
            print("No solution found.")

        return sol, job_operations, machine_operations

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
                          self._self_name, 1, sol.get_objective_value())
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
