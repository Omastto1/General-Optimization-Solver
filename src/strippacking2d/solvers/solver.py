from docplex.cp.model import CpoModel
from collections import namedtuple

from ...common.solver import CPSolver


class StripPacking2DCPSolver(CPSolver):
    def solve(self, instance, validate=False, visualize=False, force_execution=False):
        if not force_execution and len(instance._run_history) > 0:
            if instance.skip_on_optimal_solution():
                return None, None

        # Create a new CP model
        model = CpoModel()
        model.set_parameters(params=self.params)

        # Create interval variables for each rectangle's horizontal and vertical positions
        X = [model.interval_var(start=(0, instance.strip_width - instance.rectangles[i][0]), size=instance.rectangles[i][0]) for i in range(instance.no_elements)]
        Y = [model.interval_var(size=instance.rectangles[i][1]) for i in range(instance.no_elements)]

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
        z = model.integer_var(0, sum(rect[1] for rect in instance.rectangles))

        # Add constraints linking z to the Y variables
        for i in range(instance.no_elements):
            model.add(model.end_of(Y[i]) <= z)  # Y[i].get_end()

        # Minimize the total height
        model.add(model.minimize(z))
    
        # Solve the model
        solution = model.solve()
        
        # Extract and return the solution
        if solution:
            total_height = solution.get_objective_values()[0]
            placements = [(solution.get_var_solution(X[i]).get_start(), solution.get_var_solution(Y[i]).get_start()) for i in range(len(instance.rectangles))]
            return total_height, placements, solution
        else:
            return None, None, solution

