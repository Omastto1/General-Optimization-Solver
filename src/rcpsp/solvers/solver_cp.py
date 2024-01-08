from docplex.cp.model import CpoModel

from src.common.solver import CPSolver


class RCPSPCPSolver(CPSolver):
    solver_name = "CP Default"

    def build_model(self, instance, initial_solution=None):
        model = CpoModel()
        model.set_parameters(params=self.params)

        x = [model.interval_var(size=duration, name=f"{i}") for i, duration in enumerate(
            instance.durations)]  # (4)

        if initial_solution is not None:
            self.solver_name += " Hybrid"

            stp = model.create_empty_solution()

            for i, duration in enumerate(instance.durations):
                stp.add_interval_var_solution(
                    x[i], start=initial_solution[i], end=initial_solution[i] + duration)

            model.set_starting_point(stp)

        cost = model.integer_var(0, 1000000, name="cost")

        model.add(cost == model.max([model.end_of(x[i])
                                     for i in range(instance.no_jobs)]))

        if "optimum" in instance._solution and instance._solution["optimum"] is not None:
            model.add(cost >= instance._solution["optimum"])

        model.add(model.minimize(model.max([model.end_of(x[i])
                                            for i in range(instance.no_jobs)])))  # (1)

        model.add([model.sum(model.pulse(x[i], instance.requests[k][i]) for i in range(instance.no_jobs))
                   <= instance.renewable_capacities[k] for k in range(instance.no_renewable_resources)])  # (2)

        model.add([model.end_before_start(x[i], x[successor - 1]) for (i, job_successors)
                   in enumerate(instance.successors) for successor in job_successors])  # (3)

        return model, {"x": x}

    def _export_solution(self, instance, sol, model_variables):
        x = model_variables['x']

        export = []
        for i in range(instance.no_jobs):
            interval_value = sol[x[i]]
            start = interval_value.start
            end = interval_value.end

            export.append(
                {"start": start,
                 "end": end,
                 "name": x[i].name}
            )

        return {"tasks_schedule": export}

    def _solve(self, instance, validate=False, visualize=False, force_execution=False, initial_solution=None, update_history=True):
        print("Building model")
        model, model_variables = self.build_model(instance, initial_solution)

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
                is_valid = instance.validate(
                    None, None, model_variables_export)
                if is_valid:
                    print("Solution is valid.")
                else:
                    print("Solution is invalid.")
            except AssertionError as e:
                print("Solution is invalid.")
                print(e)
                return None, None

        if visualize:
            instance.visualize(model_variables_export)

        obj_value = sol.get_objective_value()
        print('Objective value:', obj_value)

        if sol.get_solve_status() == 'Optimal':
            print("Optimal solution found")
        elif sol.get_solve_status() == 'Feasible':
            print("Feasible solution found")
        else:
            print("Unknown solution status")
            print(sol.get_solve_status())

        print(sol.solution.get_objective_bounds())
        print(sol.solution.get_objective_gaps())
        print(sol.solution.get_objective_values())

        obj_value = sol.get_objective_values()[0]
        print('Objective value:', obj_value)
        instance.compare_to_reference(obj_value)

        self.add_run_to_history(instance, sol)

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
