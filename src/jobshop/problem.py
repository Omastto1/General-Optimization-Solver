from src.common.optimization_problem import OptimizationProblem
import docplex.cp.utils_visu as visu


class JobShop(OptimizationProblem):
    def __init__(self, benchmark_name, instance_name, data, solution, run_history) -> None:
        super().__init__(benchmark_name, instance_name, "JOBSHOP", data, solution, run_history)

        self.no_jobs = self._data["no_jobs"]
        self.no_machines = self._data["no_machines"]
        self.durations = self._data["durations"]
        self.machines = self._data["machines"]

    def validate(self, jobs_assignment):                
        for jobs in jobs_assignment:
            for job_subtask_no in range(1, self.no_machines):
                assert jobs[job_subtask_no-1]['end'] <= jobs[job_subtask_no]['start'], f"Job {job_subtask_no-1} ends after job {job_subtask_no} starts."

        return True

    def visualize(self, job_operations_export, machine_operations_export):
        if visu.is_visu_enabled():            
            # visu.timeline('Solution for job-shop ' +
            #               self._instance_name, 1, sol.get_objective_value())
            visu.panel('Jobs')
            for i in range(self.no_jobs):
                # visu.sequence(name='J' + str(i),
                #               intervals=[(sol.get_var_solution(job_operations[i][j]), self.machines[i][j], 'M' + str(self.machines[i][j])) for j in
                #                          range(self.no_machines)])
                visu.sequence(name='J' + str(i),
                              intervals=[(job_operations_export[i][j]['start'], job_operations_export[i][j]['end'], self.machines[i][j], 'M' + str(self.machines[i][j])) for j in
                                         range(self.no_machines)])
            visu.panel('Machines')
            for k in range(self.no_machines):
                visu.sequence(name='M' + str(k),
                              intervals=[(machine_operations_export[k][i]['start'], machine_operations_export[k][i]['end'], k, 'J' + str(i)) for i in range(self.no_jobs)])
            visu.show()
