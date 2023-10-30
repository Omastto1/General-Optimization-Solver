from src.common.optimization_problem import OptimizationProblem
import docplex.cp.utils_visu as visu


from src.common.optimization_problem import OptimizationProblem
import docplex.cp.utils_visu as visu
import matplotlib.pyplot as plt


class BinPacking1D(OptimizationProblem):
    def __init__(self, benchmark_name, instance_name, data, solution, run_history) -> None:
        super().__init__(benchmark_name, instance_name, "1DBINPACKING", data, solution, run_history)

        # self.no_items = len(instance.weights)
        self.bin_capacity = self._data["bin_capacity"]
        self.weights = self._data["weights"]
        self.no_items = len(self._data["weights"])

    def validate(self, instance, item_bin_pos_assignment, ):
        # The sum of item_bin_pos_assignment[i][j] for each item i should be 1
        sums_per_item = [sum(item_bin_pos_assignment[i][j] for j in range(instance.no_items)) for i in range(instance.no_items)]
        if not all(s == 1 for s in sums_per_item):
            raise ValueError("Constraint violated: The sum of item_bin_pos_assignment[i][j] for each item i should be 1")

        # The sum of weights for items in each bin should not exceed the capacity
        sums_per_bin = [sum(instance.weights[i] * item_bin_pos_assignment[i][j] for i in range(instance.no_items)) for j in range(instance.no_items)]
        if not all(s <= instance.bin_capacity * is_bin_used[j] for j, s in enumerate(sums_per_bin)):
            raise ValueError("Constraint violated: The sum of weights for items in each bin should not exceed the capacity")


    def visualize(self, item_bin_assignment):
        """visualize the solution of the problem
        expects one hot encoded item_bin_assignment

        Args:
            item_bin_assignment (List[List[int]]): one hot encoded item_bin_assignment
        """
        if not visu.is_visu_enabled():
            assert False, "Visualization not available. Please install docplex and enable visu."
        
        bin_item_assignment = [list(row) for row in zip(*item_bin_assignment)]

        # visu.reset()
        for j, used_bin in enumerate(bin_item_assignment):
            if sum(used_bin) > 0:
                visu.sequence(name="Bin_{}".format(j + 1))
                position = 0
                for i, is_in_bin in enumerate(used_bin):
                    if is_in_bin:
                        visu.interval(position, position + self.weights[i], i)
                        position += self.weights[i]
        visu.show()
