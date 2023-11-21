from src.common.optimization_problem import OptimizationProblem

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


class StripPacking2D(OptimizationProblem):
    def __init__(self, benchmark_name, instance_name, data, solution, run_history) -> None:
        super().__init__(benchmark_name, instance_name, "2DSTRIPPACKING", data, solution, run_history)

        self.no_elements = len(self._data["rectangles"])
        self.strip_width = self._data["strip_width"]
        self.rectangles = self._data["rectangles"]

    def validate(self, sol):
        # assert sol.get_objective_value(
        # ) <= self.horizon, "Project completion time exceeds horizon."

        # for i, job_successors in enumerate(self.successors):
        #     for successor in job_successors:
        #         assert sol.get_var_solution(x[i]).get_end() <= sol.get_var_solution(
        #             x[successor - 1]).get_start(), f"Job {i} ends after job {successor} starts."
        print("WARNING: No validation implemented for 2D Strip Packing")
        return True

    def visualize(self, sol, placements, total_height):
        rectangles = []
        for i, rectangle in enumerate(self.rectangles):
            x, y = placements[i]
            width, height = rectangle
            rectangles.append((x, y, width, height))

        # Create a figure and axis for plotting
        fig, ax = plt.subplots()
        ax.set_xlim([0, self.strip_width])
        ax.set_ylim([0, total_height])

        # Draw the large rectangle
        large_rect = Rectangle((0, 0), self.strip_width, total_height, edgecolor='black', facecolor='none')
        ax.add_patch(large_rect)

        # Draw the small rectangles within the large rectangle
        for i, (x, y, width, height) in enumerate(rectangles):
            print(x, y, width, height)
            # if orientations[i] == 'rotated':
            #     height, width = width, height
            rect = Rectangle((x, y), width, height, edgecolor='red', facecolor='green')
            ax.add_patch(rect)

        # Set the aspect ratio and display the plot
        ax.set_aspect('equal', 'box')
        plt.show()