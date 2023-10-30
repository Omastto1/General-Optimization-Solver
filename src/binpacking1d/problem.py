from src.common.optimization_problem import OptimizationProblem
import docplex.cp.utils_visu as visu


from src.common.optimization_problem import OptimizationProblem
import docplex.cp.utils_visu as visu
import matplotlib.pyplot as plt


class BinPacking1D(OptimizationProblem):
    def __init__(self, benchmark_name, instance_name, data, solution, run_history) -> None:
        super().__init__(benchmark_name, instance_name, "BinPacking1D", data, solution, run_history)

        # self.no_items = len(instance.weights)
        self.bin_capacity = self._data["bin_capacity"]
        self.weights = self._data["weights"]
        self.no_items = len(self._data["weights"])

    def validate(self, sol, jobs):
        raise NotImplementedError

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
