from docplex.cp.model import CpoModel

from src.common.solver import CPSolver


class MMRCPSPCPSolver(CPSolver):
    def build_model(self, instance):
        # define model
        model = CpoModel()
        model.set_parameters(params=self.params)

        # define variables
        jobs = range(instance.no_jobs)
        renewable_resources = range(instance.no_renewable_resources)
        non_renewable_resources = range(instance.no_non_renewable_resources)

        xs = [model.interval_var(name=f'task_{i}') for i in jobs]
        ys = [[model.interval_var(size=instance.durations[i][j], name=f'task_{i}_mode_{j}', optional=True) for j in range(
            instance.no_modes_list[i])] for i in jobs]

        cost = model.integer_var(0, 1000000, name="cost")

        model.add(cost == model.max(model.end_of(x) for x in xs))

        if "optimum" in instance._solution and instance._solution["optimum"] is not None:
            model.add(cost >= instance._solution["optimum"])

        model.add(model.minimize(model.max(model.end_of(x) for x in xs)))

        for i in jobs:
            model.add(model.alternative(xs[i], ys[i]))

        for k in renewable_resources:
            renewable_resources_requirements = [model.pulse(
                ys[i][j], instance.requests[k][i][j]) for i in jobs for j in range(instance.no_modes_list[i])]
            model.add(model.sum(renewable_resources_requirements)
                      <= instance.renewable_capacities[k])

        for k in non_renewable_resources:
            non_renewable_resources_requirements = [model.presence_of(
                ys[i][j]) * instance.requests[k+2][i][j] for i in jobs for j in range(instance.no_modes_list[i])]
            model.add(model.sum(non_renewable_resources_requirements)
                      <= instance.non_renewable_capacities[k])

        # define precedence constraints
        for (i, job_successors) in enumerate(instance.successors):
            model.add([model.end_before_start(xs[i], xs[successor - 1])
                      for successor in job_successors])

        return model, {"xs": xs, "ys": ys}

    def _export_solution(self, instance, sol, model_variables):
        ys = model_variables["ys"]

        jobs = range(instance.no_jobs)

        export = [{"start":  sol.get_var_solution(ys[i][j]).get_start(), "end":  sol.get_var_solution(ys[i][j]).get_end(), "name": ys[i][j].get_name(), "mode": j} for i in jobs for j in range(
            instance.no_modes_list[i]) if sol.get_var_solution(ys[i][j]).is_present()]

        return {"task_mode_assignment": export}

    def _solve(self, instance, validate=False, visualize=False, force_execution=False, update_history=True):
        print("Building model")
        model, model_variables = self.build_model(instance)

        print("Looking for solution")
        sol = model.solve()

        if sol.get_solve_status() in ["Unknown", "Infeasible", "JobFailed", "JobAborted"]:
            print('No solution found')
            return None, None, sol

        model_variables_export = self._export_solution(
            instance, sol, model_variables)

        if validate:
            try:
                print("Validating solution...")
                is_valid = instance.validate(model_variables_export)
                if is_valid:
                    print("Solution is valid.")
                else:
                    print("Solution is invalid.")
            except AssertionError as e:
                print("Solution is invalid.")
                print(e)
                return None, None, sol

        if visualize:
            instance.visualize(model_variables_export)

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

        if update_history:
            self.add_run_to_history(instance, sol)

        return obj_value, model_variables_export, sol
