# Draw solution
import docplex.cp.utils_visu as visu


def visualize_jobshop(instance, sol, job_operations, machine_operations):
    if sol and visu.is_visu_enabled():
        print(1, sol.get_objective_value())
        visu.timeline('Solution for job-shop ' + instance._instance_name, 1, sol.get_objective_value())
        visu.panel('Jobs')
        for i in range(instance.no_jobs):
            visu.sequence(name='J' + str(i),
                        intervals=[(sol.get_var_solution(job_operations[i][j]), instance.machines[i][j], 'M' + str(instance.machines[i][j])) for j in
                                    range(instance.no_machines)])
        visu.panel('Machines')
        for k in range(instance.no_machines):
            visu.sequence(name='M' + str(k),
                        intervals=[(sol.get_var_solution(machine_operations[k][i]), k, 'J' + str(i)) for i in range(instance.no_jobs)])
        visu.show()