from docplex.cp.model import CpoModel
from collections import namedtuple

from ...common.solver import CPSolver


class JobShopCPSolver(CPSolver):
    def _solve(self, instance, validate=False, visualize=False, force_execution=False):
        if not force_execution and len(instance._run_history) > 0:
            if instance.skip_on_optimal_solution():
                return None, None

        model = CpoModel()
        model.set_parameters(params=self.params)

        job_operations = [[model.interval_var(name=f"J_{job}_{order_index}", size=instance.durations[job][order_index])
                           for order_index in range(instance.no_machines)] for job in range(instance.no_jobs)]
        
        cost = model.integer_var(0, 1000000, name="cost")
        
        model.add(cost == model.max(model.end_of(job_operations[i][instance.no_machines-1]) for i in range(instance.no_jobs)))
        
        if "optimum" in instance._solution and instance._solution["optimum"] is not None:
            model.add(cost >= instance._solution["optimum"])
        
        model.add(
            [model.minimize(model.max(model.end_of(job_operations[i][instance.no_machines-1]) for i in range(instance.no_jobs)))] +
            [model.end_before_start(job_operations[i][j-1], job_operations[i][j])
             for i in range(instance.no_jobs) for j in range(instance.no_machines) if 0 < j]
        )

        # [ model.no_overlap( job_operations[job][machine_index] for job in range(instance.no_jobs) for machine_index in range(instance.no_machines) if instance.machines[job][machine_index] == k ) for k in range(instance.no_machines) ] +
        machine_operations = [[] for m in range(instance.no_machines)]
        for j in range(instance.no_jobs):
            for s in range(instance.no_machines):
                machine_operations[instance.machines[j]
                                   [s]].append(job_operations[j][s])
        for mops in machine_operations:
            model.add(model.no_overlap(mops))

        print("Using 10 second time limit")
        sol = model.solve()

        if sol:
            if validate:
                try:
                    print("Validating solution...")
                    instance.validate(sol, job_operations)
                    print("Solution is valid.")
                except AssertionError as e:
                    print("Solution is invalid.")
                    print(e)
                    return None, None, None

            if visualize:
                instance.visualize(sol, job_operations, machine_operations)

            print("Project completion time:", sol.get_objective_values()[0])
        else:
            print("No solution found.")
            
        # print solution
        if sol.get_solve_status() == 'Optimal':
            print("Optimal solution found")
        elif sol.get_solve_status() == 'Feasible':
            print("Feasible solution found")
        else:
            print("Unknown solution status")
            print(sol.get_solve_status())

        obj_value = sol.get_objective_values()[0]
        print('Objective value:', obj_value)
        instance.compare_to_reference(obj_value)

        Solution = namedtuple("Solution", ['job_operations', 'machine_operations'])
        variables = Solution(job_operations, machine_operations)

        instance.update_run_history(sol, variables, "CP", self.params)

        return sol, variables
