from docplex.cp.model import CpoModel
from collections import namedtuple

from ...common.solver import CPSolver


class StripPacking2DCPSolver(CPSolver):
    def build_model(self, instance):
        # Create a new CP model
        model = CpoModel()
        model.set_parameters(params=self.params)

        # Number of rectangles
        n = len(instance.rectangles)

        # Create interval variables for each rectangle's horizontal and vertical positions for both orientations
        X_main = [model.interval_var() for i in range(instance.no_elements)]
        X = [model.interval_var(start=(0, instance.strip_width - instance.rectangles[i]['width']), size=instance.rectangles[i]['width'], optional=True) for i in range(instance.no_elements)]
        X_rot = [model.interval_var(start=(0, instance.strip_width - instance.rectangles[i]['height'] if instance.strip_width - instance.rectangles[i]['height'] >= 0 else 2 ** 31), size=instance.rectangles[i]['height'], optional=True) for i in range(instance.no_elements)]
        
        print("1")
        Y_main = [model.interval_var() for i in range(instance.no_elements)]
        Y = [model.interval_var(size=instance.rectangles[i]['height'], optional=True) for i in range(instance.no_elements)]
        Y_rot = [model.interval_var(size=instance.rectangles[i]['width'], optional=True) for i in range(instance.no_elements)]

        print("2")
        # Orientation decision variables
        O = [model.binary_var() for i in range(instance.no_elements)]

        # Adjust size and domain of interval variables based on orientation
        for i in range(instance.no_elements):
            print(i)
            model.add(model.alternative(X_main[i], [X[i], X_rot[i]]))
            model.add(model.alternative(Y_main[i], [Y[i], Y_rot[i]]))

            model.add(model.if_then(O[i] == 0, model.all_of([model.presence_of(X_rot[i]), model.presence_of(Y_rot[i])])))
            model.add(model.if_then(O[i] == 1, model.all_of([model.presence_of(X[i]), model.presence_of(Y[i])])))

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
            print(i)
            for j in range(i + 1, n):
                double_no_overlap(model, X[i], X[j], Y[i], Y[j])
                double_no_overlap(model, X_rot[i], X[j], Y_rot[i], Y[j])
                double_no_overlap(model, X[i], X_rot[j], Y[i], Y_rot[j])
                double_no_overlap(model, X_rot[i], X_rot[j], Y_rot[i], Y_rot[j])

        print("3")
        # Create variable z for the total height of the packing
        z = model.integer_var(0, sum(max(rect.values()) for rect in instance.rectangles))

        print("4")
        # Add constraints linking z to the Y variables
        for i in range(instance.no_elements):
            # model.add(model.max(model.end_of(Y[i]), model.end_of(Y_rot[i])) <= z)
            model.add(model.end_of(Y_main[i]) <= z)

        # Minimize the total height
        model.add(model.minimize(z))

        return model, X, Y, X_rot, Y_rot

    def _solve(self, instance, validate=False, visualize=False, force_execution=False):
        print("Building model")
        model, X, Y, X_rot, Y_rot = self.build_model(instance)

        print("Looking for solution")
        # Solve the model
        solution = model.solve()

        if solution.get_solve_status() in ["Unknown", "Infeasible", "JobFailed", "JobAborted"]:
            print('No solution found')
            return None, None, solution
        
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
            
        self.add_run_to_history(instance, solution)
        return total_height, placements, orientations, solution