from docplex.cp.model import CpoModel

from src.common.solver import CPSolver


class StripPacking2DCPSolver(CPSolver):
    solver_name = 'CP Default Not Oriented'
    
    def build_model(self, instance, initial_solution=None):
        # Create a new CP model
        model = CpoModel(name='2D Strip Packing Not Oriented')
        model.set_parameters(params=self.params)

        # Create interval variables for each rectangle's horizontal and vertical positions
        X = [model.interval_var(start=(0, instance.strip_width - instance.rectangles[i]['width']), size=instance.rectangles[i]['width']) for i in range(instance.no_elements)]
        Y = [model.interval_var(size=instance.rectangles[i]['height']) for i in range(instance.no_elements)]

        if initial_solution is not None:
            self.solver_name += " Hybrid"
            
            stp = model.create_empty_solution()

            for i, rectangle in enumerate(instance.rectangles):
                x, y = initial_solution[i]
                width, height = rectangle['width'], rectangle['height']
                stp.add_interval_var_solution(X[i], start=x, end=x + width)
                stp.add_interval_var_solution(Y[i], start=y, end=y + height)

            model.set_starting_point(stp)

        # Add non-overlap constraints
        for i in range(instance.no_elements):
            for j in range(i + 1, instance.no_elements):
                # model.add(model.no_overlap([X[i], X[j]]))
                # model.add(model.no_overlap([Y[i], Y[j]]))

                # Non-overlapping conditions
                no_overlap_X1 = model.end_of(X[i]) <= model.start_of(X[j])
                no_overlap_X2 = model.end_of(X[j]) <= model.start_of(X[i])
                no_overlap_Y1 = model.end_of(Y[i]) <= model.start_of(Y[j])
                no_overlap_Y2 = model.end_of(Y[j]) <= model.start_of(Y[i])

                # Ensuring that at least one non-overlapping condition is satisfied
                model.add((no_overlap_X1 | no_overlap_X2) | (no_overlap_Y1 | no_overlap_Y2))

        # Create variable z for the total height of the packing
        z = model.integer_var(0, sum(rect['height'] for rect in instance.rectangles))

        # Add constraints linking z to the Y variables
        for i in range(instance.no_elements):
            model.add(model.end_of(Y[i]) <= z)  # Y[i].get_end()

        # Minimize the total height
        model.add(model.minimize(z))

        return model, X, Y
    
    def _export_solution(self, instance, solution, X, Y):
        placements = [(solution.get_var_solution(X[i]).get_start(), solution.get_var_solution(Y[i]).get_start()) for i in range(len(instance.rectangles))]

        return placements

    def _solve(self, instance, validate=False, visualize=False, force_execution=False, initial_solution=None, update_history=True):
        if not force_execution and len(instance._run_history) > 0:
            if instance.skip_on_optimal_solution():
                return None, None

        model, X, Y = self.build_model(instance, initial_solution)
    
        # Solve the model
        solution = model.solve()

        if solution.get_solve_status() in ["Unknown", "Infeasible", "JobFailed", "JobAborted"]:
            print('No solution found')
            return None, None, solution
        
        placements = self._export_solution(instance, solution, X, Y)
        
        total_height = solution.get_objective_values()[0]

        if validate:
            try:
                print("Validating solution...")
                is_valid = instance.validate(solution)
                if is_valid:
                    print("Solution is valid.")
                else:
                    print("Solution is invalid.")
            except AssertionError as e:
                print("Solution is invalid.")
                print(e)
                return None, None
        
        if visualize:
            instance.visualize(solution, placements, total_height)

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

        return total_height, placements, solution

