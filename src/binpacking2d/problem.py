import random
from src.common.optimization_problem import OptimizationProblem
import docplex.cp.utils_visu as visu


class BinPacking2D(OptimizationProblem):
    def __init__(self, benchmark_name, instance_name, data, solution, run_history) -> None:
        super().__init__(benchmark_name, instance_name,
                         "2DBINPACKING", data, solution, run_history)

        self.bin_size = self._data["bin_size"]
        self.items_sizes = self._data["items_sizes"]
        self.no_items = len(self._data["items_sizes"])

    def validate(self, sol, jobs):
        raise NotImplementedError

    def visualize(self, assignment, orientations):
        if not visu.is_visu_enabled():
            print("Visualization not available. Please install docplex and enable visu.")
            return

        import numpy as np
        transposed_arr = np.transpose(assignment, (2, 1, 0))

        visu.timeline('Solution for RCPSP ')  # + filename)
        for k, used_bin in enumerate(transposed_arr):
            if sum(sum(row) for row in used_bin) > 0:
                visu.panel("Bin {}".format(k + 1))
                # Add a dummy interval to set the boundaries
                visu.interval(0, self.bin_size[0])

                for i in range(len(used_bin)):
                    for j, is_in_bin in enumerate(used_bin[i]):
                        if is_in_bin:
                            x_coord = i % self.bin_size[0]
                            y_coord = i // self.bin_size[0]
                            print(x_coord, y_coord)
                            rect_width = self.items_sizes[j][0] if orientations[j] == 0 else self.items_sizes[j][1]
                            rect_height = self.items_sizes[j][1] if orientations[j] == 0 else self.items_sizes[j][0]

                            segment_style = {
                                "color": random_color(),
                                "label": f"Rectangle {j + 1}"
                            }
                            # Use visu.function to visualize the rectangle
                            visu.function(segments=[
                                          (x_coord, x_coord + rect_width, y_coord, y_coord + rect_height)], style=segment_style, color=random_color())

        visu.show()


def random_color():
    return (random.random(), random.random(), random.random())
