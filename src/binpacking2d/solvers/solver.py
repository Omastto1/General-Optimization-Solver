from docplex.cp.model import CpoModel

from ...common.solver import CPSolver

class BinPacking2DCPSolver(CPSolver):
    def solve(self, instance, validate=False, visualize=False, force_execution=False):
        if not force_execution and len(instance._run_history) > 0:
            if instance.skip_on_optimal_solution():
                return None, None

        model = CpoModel(name="2DBinPacking")
        model.set_parameters(params=self.params)
        
        # Decision variables
        x = [[[model.binary_var(name="x_{}_{}_{}".format(i, j, k)) for k in range(instance.no_items)] for j in range(instance.bin_size[0] * instance.bin_size[1])] for i in range(instance.no_items)]
        orientation = [model.binary_var(name="orientation_{}".format(i)) for i in range(instance.no_items)]
        used = [model.binary_var(name="used_{}".format(k)) for k in range(instance.no_items)]
        
        # Each rectangle should be in exactly one bin and position
        for i in range(instance.no_items):
            model.add(model.sum(x[i][j][k] for j in range(instance.bin_size[0] * instance.bin_size[1]) for k in range(instance.no_items)) == 1)
        
        # No overlapping and fit within bin boundaries (simplified representation)

        for bin_no in range(instance.no_items):
            for rectangle_no in range(instance.no_items):
                for pos in range(instance.bin_size[0] * instance.bin_size[1]):
                    # Get coordinates from the flattened index pos
                    x_coord = pos % instance.bin_size[0]
                    y_coord = pos // instance.bin_size[0]
                    # Constraints ensuring rectangles do not exceed bin boundaries
                    model.add((x_coord + instance.items_sizes[rectangle_no][0]) * x[rectangle_no][pos][bin_no] <= instance.bin_size[0])
                    model.add((y_coord + instance.items_sizes[rectangle_no][1]) * x[rectangle_no][pos][bin_no] <= instance.bin_size[1])
                    
                    # Constraints for rotated rectangles
                    model.add((x_coord + instance.items_sizes[rectangle_no][1]) * x[rectangle_no][pos][bin_no] * orientation[rectangle_no] <= instance.bin_size[0])
                    model.add((y_coord + instance.items_sizes[rectangle_no][0]) * x[rectangle_no][pos][bin_no] * orientation[rectangle_no] <= instance.bin_size[1])

                    # Constraints ensuring rectangles do not overlap
                    for rectangle_no2 in range(instance.no_items):
                        for pos2 in range(instance.bin_size[0] * instance.bin_size[1]):
                            if rectangle_no != rectangle_no2:  #  and pos != pos2:
                                x_coord2 = pos2 % instance.bin_size[0]
                                y_coord2 = pos2 // instance.bin_size[0]

                                if x_coord + instance.items_sizes[rectangle_no][0] > x_coord2 \
                                    and x_coord2 + instance.items_sizes[rectangle_no2][0] > x_coord \
                                        and y_coord + instance.items_sizes[rectangle_no][1] > y_coord2 \
                                            and y_coord2 + instance.items_sizes[rectangle_no][1] > y_coord:
                                    model.add(x[rectangle_no][pos][bin_no] + x[rectangle_no2][pos2][bin_no] <= 1)

        for k in range(instance.no_items):
            for i in range(instance.no_items):
                for j in range(instance.bin_size[0] * instance.bin_size[1]):
                    model.add(used[k] >= x[i][j][k])
        
        # Objective: minimize the number of bins used
        model.add(model.minimize(model.sum(used[k] for k in range(instance.no_items))))
        
        # Solve the model
        solution = model.solve()
        
        # Extract and return the solution
        if solution:
            bins_used = sum([int(solution[used[k]]) for k in range(instance.no_items)])
            assignment = [[[int(solution[x[i][j][k]]) for k in range(instance.no_items)] for j in range(instance.bin_size[0] * instance.bin_size[1])] for i in range(instance.no_items)]
            orientations = [int(solution[orientation[i]]) for i in range(instance.no_items)]
            return bins_used, assignment, orientations
        else:
            return None, None, None
