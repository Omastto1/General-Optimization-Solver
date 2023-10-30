from docplex.cp.model import CpoModel

from ...common.solver import CPSolver

class BinPacking1DCPSolver(CPSolver):
    def solve(self, instance, validate=False, visualize=False, force_execution=False):
        if not force_execution and len(instance._run_history) > 0:
            if instance.skip_on_optimal_solution():
                return None, None

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
        
        # Solve the model
        solution = model.solve()

        if solution:
            if validate:
                try:
                    print("Validating solution...")
                    instance.validate(instance, item_bin_pos_assignment, is_bin_used)
                    print("Solution is valid.")

                    obj_value = solution.get_objective_value()
                    print("Project completion time:", obj_value)

                    instance.compare_to_reference(obj_value)
                except AssertionError as e:
                    print("Solution is invalid.")
                    print(e)
                    return None, None

            if visualize:
                bins_used = sum([int(solution[is_bin_used[j]]) for j in range(instance.no_items)])
                assignment = [[int(solution[item_bin_pos_assignment[i][j]]) for j in range(instance.no_items)] for i in range(instance.no_items)]
                instance.visualize(assignment)

            # for i in range(no_jobs):
            #     print(f"Activity {i}: start={sol[i].get_start()}, end={sol[i].get_end()}")

            print(solution.solution.get_objective_bounds())
            print(solution.solution.get_objective_gaps())
            print(solution.solution.get_objective_values())

            # obj_value = solution.objective_value
            obj_value = solution.get_objective_values()[0]
            print('Objective value:', obj_value)
            # start_times = [solution.get_var_solution(x[i]).get_start() for i in range(instance.no_jobs)]
            instance.compare_to_reference(obj_value)
        else:
            print("No solution found.")
        
        # Extract and return the solution
        if solution:
            bins_used = sum([int(solution[is_bin_used[j]]) for j in range(instance.no_items)])
            assignment = [[int(solution[item_bin_pos_assignment[i][j]]) for j in range(instance.no_items)] for i in range(instance.no_items)]
            return bins_used, assignment, solution
        else:
            return None, None, solution



