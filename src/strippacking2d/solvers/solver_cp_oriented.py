from docplex.cp.model import CpoModel

from src.common.solver import CPSolver


class StripPacking2DCPSolver(CPSolver):
    solver_name = 'CP Default Oriented'
    
    def build_model(self, instance, initial_solution=None):
        # Create a new CP model
        model = CpoModel('2D Strip Packing Oriented')
        model.set_parameters(params=self.params)

        # Create interval variables for each rectangle's horizontal and vertical positions for both orientations
        X_main = [model.interval_var() for i in range(instance.no_elements)]
        X = [model.interval_var(name=f'rectangle_{i}_X', start=(0, instance.strip_width - instance.rectangles[i]['width']), size=instance.rectangles[i]['width'], optional=True) for i in range(instance.no_elements)]
        X_rot = [model.interval_var(name=f'rectangle_{i}_Xrot', start=(0, instance.strip_width - instance.rectangles[i]['height'] if instance.strip_width - instance.rectangles[i]['height'] >= 0 else 2 ** 31), size=instance.rectangles[i]['height'], optional=True) for i in range(instance.no_elements)]
        
        Y_main = [model.interval_var() for i in range(instance.no_elements)]
        Y = [model.interval_var(name=f'rectangle_{i}_Y', size=instance.rectangles[i]['height'], optional=True) for i in range(instance.no_elements)]
        Y_rot = [model.interval_var(name=f'rectangle_{i}_Yrot', size=instance.rectangles[i]['width'], optional=True) for i in range(instance.no_elements)]

        # Orientation decision variables
        O = [model.binary_var() for i in range(instance.no_elements)]
        
        if initial_solution is not None:
            if not self.solver_name.endswith(" Hybrid"):
                self.solver_name += " Hybrid"
            
            stp = model.create_empty_solution()

            for i, rectangle in enumerate(instance.rectangles):
                x, y = initial_solution[i]
                width, height = rectangle['width'], rectangle['height']
                stp.add_interval_var_solution(X_main[i], start=x, end=x + width)
                stp.add_interval_var_solution(Y_main[i], start=y, end=y + height)
                stp.add_interval_var_solution(X[i], start=x, end=x + width)
                stp.add_interval_var_solution(Y[i], start=y, end=y + height)
                stp.add_interval_var_solution(X_rot[i], presence=False)
                stp.add_interval_var_solution(Y_rot[i], presence=False)

                stp.add_integer_var_solution(O[i], 1)

            model.set_starting_point(stp)


        # Adjust size and domain of interval variables based on orientation
        for i in range(instance.no_elements):
            # connect main variable with both orientations so that X main is scheduled (copy of one or rotation variable)
            model.add(model.alternative(X_main[i], [X[i], X_rot[i]]))
            model.add(model.alternative(Y_main[i], [Y[i], Y_rot[i]]))

            # Define that only rotated variables or not rotated variables for a specific rectangle cna be used
            model.add(model.if_then(O[i] == 0, model.all_of([model.presence_of(X_rot[i]), model.presence_of(Y_rot[i])])))
            model.add(model.if_then(O[i] == 1, model.all_of([model.presence_of(X[i]), model.presence_of(Y[i])])))

        # Add non-overlap constraints
        for i in range(instance.no_elements):
            for j in range(i + 1, instance.no_elements):
                no_overlap_X1 = model.end_of(X_main[i]) <= model.start_of(X_main[j])
                no_overlap_X2 = model.end_of(X_main[j]) <= model.start_of(X_main[i])
                no_overlap_Y1 = model.end_of(Y_main[i]) <= model.start_of(Y_main[j])
                no_overlap_Y2 = model.end_of(Y_main[j]) <= model.start_of(Y_main[i])
                
                # Ensuring that at least one non-overlapping condition is satisfied
                model.add((no_overlap_X1 | no_overlap_X2) | (no_overlap_Y1 | no_overlap_Y2))
            
        # Minimize the total height
        model.add(model.minimize(model.max([model.end_of(Y_main[i])
                                            for i in range(instance.no_elements)])))  # (1)


        return model, X, Y, X_rot, Y_rot
    
    def _export_solution(self, instance, solution, X, Y, X_rot, Y_rot):
        placements = []
        orientations = []
        for i, (w, h) in enumerate(instance.rectangles):
            print("A", i, solution.get_var_solution(X[i]).is_present(), solution.get_var_solution(Y[i]).is_present())
            if solution.get_var_solution(X[i]).is_present():
                placements.append((solution.get_var_solution(X[i]).get_start(), solution.get_var_solution(Y[i]).get_start(), "original"))
                orientations.append("original")

            else:
                placements.append((solution.get_var_solution(X_rot[i]).get_start(), solution.get_var_solution(Y_rot[i]).get_start(), "rotated"))
                orientations.append("rotated")
        
        return placements, orientations

    def _solve(self, instance, validate=False, visualize=False, force_execution=False, initial_solution=None, update_history=True):
        print("Building model")
        model, X, Y, X_rot, Y_rot = self.build_model(instance, initial_solution)

        print("Looking for solution")
        # Solve the model
        solution = model.solve()

        if solution.get_solve_status() in ["Unknown", "Infeasible", "JobFailed", "JobAborted"]:
            print('No solution found')
            return None, None, solution
        
        placements, orientations = self._export_solution(instance, solution, X, Y, X_rot, Y_rot)
        total_height = solution.get_objective_value()

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
            instance.visualize(None, placements, total_height)

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
            
        return total_height, {"placements": placements, "orientations": orientations}, solution