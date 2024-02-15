from typing import List
from src.common.optimization_problem import OptimizationProblem

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


class StripPacking2D(OptimizationProblem):
    def __init__(self, benchmark_name, instance_name, data, solution, run_history) -> None:
        super().__init__(benchmark_name, instance_name,
                         "2DSTRIPPACKING", data, solution, run_history)

        self.no_elements = len(self._data["rectangles"])
        self.strip_width = self._data["strip_width"]
        self.rectangles: List[dict] = self._data["rectangles"]

    def validate(self, sol):
        print("WARNING: No validation implemented for 2D Strip Packing")
        return True

    def visualize(self, sol, placements, total_height):
        rectangles = []
        for i, rectangle in enumerate(self.rectangles):

            width, height = rectangle['width'], rectangle['height']
            if len(placements[i]) == 2:
                x, y = placements[i]
            elif len(placements[i]) == 3:
                x, y, orientation = placements[i]

                if orientation == 'rotated':
                    height, width = width, height
            else:
                raise Exception("Invalid placement format")

            rectangles.append((x, y, width, height))

        # Create a figure and axis for plotting
        fig, ax = plt.subplots()
        fig.suptitle(f'{self._benchmark_name} - {self._instance_name}')
        ax.set_xlim([0, self.strip_width])
        ax.set_ylim([0, total_height])

        # Draw the large rectangle
        large_rect = Rectangle((0, 0), self.strip_width,
                               total_height, edgecolor='black', facecolor='none')
        ax.add_patch(large_rect)

        # Draw the small rectangles within the large rectangle
        for i, (x, y, width, height) in enumerate(rectangles):
            rect = Rectangle((x, y), width, height,
                             edgecolor='red', facecolor='green')
            ax.add_patch(rect)

        # Set the aspect ratio and display the plot
        ax.set_aspect('equal', 'box')
        plt.show()
