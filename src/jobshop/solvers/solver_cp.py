from docplex.cp.model import CpoModel
from collections import namedtuple

from ...common.solver import CPSolver


class JobShopCPSolver(CPSolver):

    def build_model(self, instance):
        model = CpoModel()
        model.set_parameters(params=self.params)

        job_operations = [[model.interval_var(name=f"J_{job}_{order_index}", size=instance.durations[job][order_index])
                           for order_index in range(instance.no_machines)] for job in range(instance.no_jobs)]

        cost = model.integer_var(0, 1000000, name="cost")

        model.add(cost == model.max(model.end_of(
            job_operations[i][instance.no_machines-1]) for i in range(instance.no_jobs)))

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

        return model, job_operations  # , machine_operations

    def _export_solution(self, instance, sol, job_operations):
        job_operations_export = []
        for i in range(instance.no_jobs):
            job_operations_export.append([])
            for j in range(instance.no_machines):
                interval_value = sol[job_operations[i][j]]
                machine = instance.machines[i][j]
                start = interval_value.start
                end = interval_value.end

                job_operations_export[i].append(
                    {"start": start,
                     "end": end,
                     "machine": machine}
                )

        return job_operations_export

    def _solve(self, instance, validate=False, visualize=False, force_execution=False):
        print("Building model")
        model, job_operations = self.build_model(
            instance)  # , machine_operations

        print("Looking for solution")
        sol = model.solve()

        if sol.get_solve_status() in ["Unknown", "Infeasible", "JobFailed", "JobAborted"]:
            print('No solution found')
            return None, None, sol
        
        job_operations_export = self._export_solution(
            instance, sol, job_operations)

        if validate:
            try:
                print("Validating solution...")
                is_valid = instance.validate(job_operations_export)
                if is_valid:
                    print("Solution is valid.")
                else:
                    print("Solution is invalid.")
            except AssertionError as e:
                print("Solution is invalid.")
                print(e)
                return None, None, None

        if visualize:
            instance.visualize(job_operations_export)

        obj_value = sol.get_objective_values()[0]
        print('Objective value:', obj_value)

        if sol.get_solve_status() == 'Optimal':
            print("Optimal solution found")
        elif sol.get_solve_status() == 'Feasible':
            print("Feasible solution found")
        else:
            print("Unknown solution status")
            print(sol.get_solve_status())

        instance.compare_to_reference(obj_value)

        self.add_run_to_history(instance, sol)

        return obj_value, {"jobs": job_operations_export}, sol
