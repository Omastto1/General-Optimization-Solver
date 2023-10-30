from src.strippacking2d.problem import StripPacking2D
from src.strippacking2d.solver import StripPacking2DSolver
import docplex.cp.utils_visu as visu
import matplotlib.pyplot as plt



# Example usage
rectangles = [(3, 4), (5, 6), (2, 5)]  # Each tuple represents (width, height)
strip_width = 7

problem = StripPacking2D(benchmark_name="StripPacking2DTest", instance_name="Test01", data={"rectangles": rectangles, "strip_width": strip_width}, solution={}, run_history={})


total_height, placements, solution = StripPacking2DSolver()._solve_cp(problem, validate=False, visualize=False, force_execution=True)

print("Total height:", total_height)
print("Placement of rectangles:", placements)

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

rectangles = []
for i, rectangle in enumerate(problem.rectangles):
    x, y = placements[i]
    width, height = rectangle
    rectangles.append((x, y, width, height))

# Create a figure and axis for plotting
fig, ax = plt.subplots()
ax.set_xlim([0, problem.strip_width])
ax.set_ylim([0, total_height])

# Draw the large rectangle
large_rect = Rectangle((0, 0), problem.strip_width, total_height, edgecolor='black', facecolor='none')
ax.add_patch(large_rect)

# Draw the small rectangles within the large rectangle
for x, y, width, height in rectangles:
    print(x, y, width, height)
    rect = Rectangle((x, y), width, height, edgecolor='red', facecolor='green')
    ax.add_patch(rect)

# Set the aspect ratio and display the plot
ax.set_aspect('equal', 'box')
plt.show()



# seems easy but has visu.rectangle which does not exist
# if solution:
#     # Create a new visualization
#     visu.timeline("Strip Packing Solution", 0, strip_width)
    
#     for i, ((x, y), (w, h)) in enumerate(zip(placements, rectangles)):
#         # orientation = "original" if solution[O[i]] == 0 else "rotated"
        
#         # if orientation == "original":
#         visu.rectangle(x, y, 
#                         w, h, color="blue", legend=f"Rectangle {i}")
#         # else:
#         #     visu.rectangle(solution.get_value(X_rot[i].get_start()), solution.get_value(Y_rot[i].get_start()), 
#         #                    h, w, color="red", legend=f"Rectangle {i} (rotated)")
    
#     # Show the visualization
#     visu.show()
# else:
#     print("No solution found.")

