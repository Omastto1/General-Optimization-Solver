from docplex.cp.model import CpoModel
from collections import namedtuple

from ...common.solver import CPSolver


class StripPacking2DCPSolver(CPSolver):
    def _solve(self, instance, validate=False, visualize=False, force_execution=False):
        if not force_execution and len(instance._run_history) > 0:
            if instance.skip_on_optimal_solution():
                return None, None

        # Create a new CP model
        model = CpoModel()
        model.set_parameters(params=self.params)

        # Number of rectangles
        n = len(instance.rectangles)

        # Create interval variables for each rectangle's horizontal and vertical positions for both orientations
        X_main = [model.interval_var() for i in range(instance.no_elements)]
        X = [model.interval_var(start=(0, instance.strip_width - instance.rectangles[i][0]), size=instance.rectangles[i][0], optional=True) for i in range(instance.no_elements)]
        X_rot = [model.interval_var(start=(0, instance.strip_width - instance.rectangles[i][1]), size=instance.rectangles[i][1], optional=True) for i in range(instance.no_elements)]
        
        Y_main = [model.interval_var() for i in range(instance.no_elements)]
        Y = [model.interval_var(size=instance.rectangles[i][1], optional=True) for i in range(instance.no_elements)]
        Y_rot = [model.interval_var(size=instance.rectangles[i][0], optional=True) for i in range(instance.no_elements)]

        # Orientation decision variables
        O = [model.binary_var() for i in range(instance.no_elements)]

        # Adjust size and domain of interval variables based on orientation
        for i in range(instance.no_elements):
            model.add(model.alternative(X_main[i], [X[i], X_rot[i]]))
            model.add(model.alternative(Y_main[i], [Y[i], Y_rot[i]]))

            # model.add(model.if_then(X[i].size[0] > 0, X_rot[i].size[0] == 0))
            # model.add(model.if_then(X_rot[i].size[0] > 0, X[i].size[0] == 0))

            # model.add(model.if_then(Y[i].size[0] > 0, Y_rot[i].size[0] == 0))
            # model.add(model.if_then(Y_rot[i].size[0] > 0, Y[i].size[0] == 0))

            model.add(model.if_then(O[i] == 0, model.all_of([model.presence_of(X_rot[i]), model.presence_of(Y_rot[i])])))
            model.add(model.if_then(O[i] == 1, model.all_of([model.presence_of(X[i]), model.presence_of(Y[i])])))
            # model.add(model.if_then(O[i] == 0, model.all_of([X_rot[i].is_present(), Y_rot[i].is_present()])))
            # model.add(model.if_then(O[i] == 1, model.all_of([X[i].is_present(), Y[i].is_present()])))
            # model.add(model.logical_and(O[i] == 0, Y_main[i].is_present()))

            # model.add(model.alternative(X_main[i], [X[i], X_rot[i]], O[i]))  # 
            # model.add(model.alternative(Y_main[i], [Y[i], Y_rot[i]], O[i]))  # 

            # model.add(model.logical_and(O[i] == 0, X_main[i].is_present()))
            # model.add(model.logical_and(O[i] == 0, Y_main[i].is_present()))

            # model.add(model.logical_or(model.logical_and(Y[i].is_present(), X[i].is_present())), model.logical_and(Y_rot[i].is_present(), X_rot[i].is_present()))
    
            #(O[i] = 1 implies X_rot[i] and Y_rot[i] are selected
            # model.add(model.if_then(O[i] == 1, X[i].is_present()))
            # model.add(model.if_then(O[i] == 1, Y[i].is_present()))
            # model.add(model.if_then(O[i] == 0, X_rot[i].is_present()))
            # model.add(model.if_then(O[i] == 0, Y_rot[i].is_present()))

            
            # model.add(model.if_then(X[i].is_present(), Y[i].is_present()))
            # model.add(model.if_then(Y[i].is_present(), X[i].is_present()))
            # model.add(model.if_then(X_rot[i].is_present(), Y_rot[i].is_present()))
            # model.add(model.if_then(Y_rot[i].is_present(), X_rot[i].is_present()))

        def double_no_overlap(model, X1, X2, Y1, Y2):
            # start before end
            no_overlap_X1 = model.end_of(X1) <= model.start_of(X2)
            no_overlap_X2 = model.end_of(X2) <= model.start_of(X1)
            no_overlap_Y1 = model.end_of(Y1) <= model.start_of(Y2)
            no_overlap_Y2 = model.end_of(Y2) <= model.start_of(Y1)

            # Ensuring that at least one non-overlapping condition is satisfied
            model.add((no_overlap_X1 | no_overlap_X2) | (no_overlap_Y1 | no_overlap_Y2))


        # Add non-overlap constraints considering both orientations
        for i in range(instance.no_elements):
            for j in range(i + 1, n):
                double_no_overlap(model, X[i], X[j], Y[i], Y[j])
                double_no_overlap(model, X_rot[i], X[j], Y_rot[i], Y[j])
                double_no_overlap(model, X[i], X_rot[j], Y[i], Y_rot[j])
                double_no_overlap(model, X_rot[i], X_rot[j], Y_rot[i], Y_rot[j])

        # Create variable z for the total height of the packing
        z = model.integer_var(0, sum(max(rect) for rect in instance.rectangles))

        # Add constraints linking z to the Y variables
        for i in range(instance.no_elements):
            # model.add(model.max(model.end_of(Y[i]), model.end_of(Y_rot[i])) <= z)
            model.add(model.end_of(Y_main[i]) <= z)

        # Minimize the total height
        model.add(model.minimize(z))

            # Solve the model
        solution = model.solve()
        
        # Extract and return the solution
        if solution:
            total_height = solution.get_objective_values()[0]
            placements = []
            orientations = []
            for i, (w, h) in enumerate(instance.rectangles):
                if solution.get_var_solution(X[i]).is_present():
                    placements.append((solution.get_var_solution(X[i]).get_start(), solution.get_var_solution(Y[i]).get_start()))
                    orientations.append("original")
                else:
                    placements.append((solution.get_var_solution(X_rot[i]).get_start(), solution.get_var_solution(Y_rot[i]).get_start()))
                    orientations.append("rotated")
            return total_height, placements, orientations, solution
        else:
            return None, None, None, solution