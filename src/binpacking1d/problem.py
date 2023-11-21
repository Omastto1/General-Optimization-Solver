from src.common.optimization_problem import OptimizationProblem
import docplex.cp.utils_visu as visu
from typing import List


class BinPacking1D(OptimizationProblem):
    def __init__(self, benchmark_name, instance_name, data, solution, run_history) -> None:
        super().__init__(benchmark_name, instance_name, "1DBINPACKING", data, solution, run_history)

        # self.no_items = len(instance.weights)
        self.bin_capacity = self._data["bin_capacity"]
        self.weights = self._data["weights"]
        self.no_items = len(self._data["weights"])

    def validate(self, model_variables):
        """_summary_

        Args:
            item_bin_pos_assignment (_type_): one hot encoded item_bin_assignment - item_bin_assignment[0][5] - is the first item in the sixth bin
            is_bin_used (List[int]): 0/1 list showing which bin is used (=1)

        Raises:
            ValueError: sum of item weights in one of bins exceeds the capacity

        Returns:
            bool: True if valid
        """
        item_bin_pos_assignment = model_variables['item_bin_pos_assignment']
        is_bin_used = model_variables['is_bin_used']

        # The sum of weights for items in each bin should not exceed the capacity
        sums_per_bin = [sum(self.weights[i] * item_bin_pos_assignment[i][j] for i in range(self.no_items)) for j in range(self.no_items)]
        if not all(s <= self.bin_capacity * is_bin_used[j] for j, s in enumerate(sums_per_bin)):
            raise ValueError("Constraint violated: The sum of weights for items in each bin should not exceed the capacity")
    
        return True


    def visualize(self, model_variables):
        """visualize the solution of the problem
        expects one hot encoded item_bin_assignment

        Args:
            item_bin_assignment (List[List[int]]): one hot encoded item_bin_assignment - item_bin_assignment[0][5] - is the first item in the sixth bin
        """
        item_bin_assignment = model_variables['item_bin_pos_assignment']
        
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
