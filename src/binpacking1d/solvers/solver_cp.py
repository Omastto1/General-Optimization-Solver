from docplex.cp.model import CpoModel

from src.common.solver import CPSolver


class BinPacking1DCPSolver(CPSolver):
    solver_name = 'CP Default'
    
    def build_model(self, instance):
        model = CpoModel(name="BinPacking")
        model.set_parameters(params=self.params)

        # Decision variables
        item_bin_pos_assignment = [[model.binary_var(name="x_{}_{}".format(i, j)) for j in range(instance.no_items)] for i in range(instance.no_items)]
        is_bin_used = [model.binary_var(name="used_{}".format(j)) for j in range(instance.no_items)]
        
        # Each item should be in exactly one bin
        for i in range(instance.no_items):
            model.add(model.sum(item_bin_pos_assignment[i][j] for j in range(instance.no_items)) == 1)
        
        # The sum of weights for items in each bin should not exceed the capacity
        for j in range(instance.no_items):
            model.add(model.sum(instance.weights[i] * item_bin_pos_assignment[i][j] for i in range(instance.no_items)) <= instance.bin_capacity * is_bin_used[j])
        
        # Objective: minimize the number of bins used
        model.add(model.minimize(model.sum(is_bin_used[j] for j in range(instance.no_items))))

        return model, {"item_bin_pos_assignment": item_bin_pos_assignment, "is_bin_used": is_bin_used}, 
    
    def _export_solution(self, instance, solution, model_variables):
        item_bin_pos_assignment = model_variables['item_bin_pos_assignment']
        is_bin_used = model_variables['is_bin_used']

        item_bin_pos_assignment_export = [[solution[item_bin_pos_assignment[i][j]] for j in range(instance.no_items)] for i in range(instance.no_items)]
        is_bin_used_export = [solution[is_bin_used[j]] for j in range(instance.no_items)]
        
        return {"item_bin_pos_assignment": item_bin_pos_assignment_export, "is_bin_used": is_bin_used_export}

    def _solve(self, instance, validate=False, visualize=False, force_execution=False, update_history=True):
        print("Building model")
        model, model_variables = self.build_model(instance)
        
        print("Looking for solution")
        solution = model.solve()

        if solution.get_solve_status() in ["Unknown", "Infeasible", "JobFailed", "JobAborted"]:
            print('No solution found')
            return None, None, solution
        
        model_variables_export = self._export_solution(instance, solution, model_variables)

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
                return None, None

        if visualize:
            instance.visualize(model_variables_export)

        obj_value = solution.get_objective_value()
        print('Objective value:', obj_value)

        if solution.get_solve_status() == 'Optimal':
            print("Optimal solution found")
        elif solution.get_solve_status() == 'Feasible':
            print("Feasible solution found")
        else:
            print("Unknown solution status")
            print(solution.get_solve_status())

        print(solution.solution.get_objective_bounds())
        print(solution.solution.get_objective_gaps())
        print(solution.solution.get_objective_values())

        instance.compare_to_reference(obj_value)

        if update_history:
            self.add_run_to_history(instance, solution)
        
        return obj_value, model_variables_export, solution



