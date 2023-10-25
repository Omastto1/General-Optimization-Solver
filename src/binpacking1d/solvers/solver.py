

from docplex.cp.model import CpoModel

from ...common.solver import Solver

class BinPacking1DSolver(Solver):
    def _solve_cp(self, instance, validate=False, visualize=False, force_execution=False):
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
        
        # Extract and return the solution
        if solution:
            bins_used = sum([int(solution[is_bin_used[j]]) for j in range(instance.no_items)])
            assignment = [[int(solution[item_bin_pos_assignment[i][j]]) for j in range(instance.no_items)] for i in range(instance.no_items)]
            return bins_used, assignment
        else:
            return None, None



