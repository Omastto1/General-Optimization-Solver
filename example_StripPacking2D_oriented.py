from src.strippacking2doriented.problem import StripPacking2D
# from src.strippacking2doriented.solver import StripPacking2DSolver
from src.strippacking2d.solver import StripPacking2DSolver
import docplex.cp.utils_visu as visu
import matplotlib.pyplot as plt


# Example usage
rectangles = [(3, 4), (5, 6), (2, 5)]  # Each tuple represents (width, height)
strip_width = 7

def parse_txt_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Extracting number of items and strip width
    n = int(lines[0].strip())
    W = int(lines[1].strip())

    items = []

    # Extracting items details
    for line in lines[2:]:
        index, width, height = map(int, line.strip().split())
        items.append([width, height])

    return W, items

import json
def parse_bkf_benchmark(file_path):
    asd = json.loads(open(file_path).read())

    W = asd['Objects'][0]['Length']
    items = [[item['Length'], item["Height"]] for item in asd['Items']]

    return W, items


results = {}

for i in range(1, 6):
    strip_width, rectangles = parse_bkf_benchmark(f"data/2DSTRIPPACKING/BKW/{i}.json")

    # strip_width, rectangles = parse_txt_file("data/2DSTRIPPACKING/zdf2.txt")

    problem = StripPacking2D(benchmark_name="StripPacking2DTest", instance_name="Test01", data={"rectangles": rectangles, "strip_width": strip_width}, solution={}, run_history={})


    total_height, placements, solution = StripPacking2DSolver(TimeLimit=300)._solve_cp(problem, validate=False, visualize=False, force_execution=True)  # orientations, 

    print("Total height:", total_height)
    print("Placement of rectangles:", placements)
    # print("orientations:", orientations)
    print("dimensions", rectangles)

    results[f'BKW{i}'] = {"Height": total_height, "Rectangles": [[*rectangle, *placement] for (rectangle, placement) in zip(rectangles, placements)]}

    # problem.visualize(solution, placements, total_height)  # orientations,

with open("cp_results.json", 'w+', encoding='utf-8') as file:
    json.dump(results, file)

# import matplotlib.pyplot as plt
# from matplotlib.patches import Rectangle

# rectangles = []
# for i, rectangle in enumerate(problem.rectangles):
#     x, y = placements[i]
#     width, height = rectangle
#     rectangles.append((x, y, width, height))

# # Create a figure and axis for plotting
# fig, ax = plt.subplots()
# ax.set_xlim([0, problem.strip_width])
# ax.set_ylim([0, total_height])

# # Draw the large rectangle
# large_rect = Rectangle((0, 0), problem.strip_width, total_height, edgecolor='black', facecolor='none')
# ax.add_patch(large_rect)

# # Draw the small rectangles within the large rectangle
# for i, (x, y, width, height) in enumerate(rectangles):
#     print(x, y, width, height)
#     # if orientations[i] == 'rotated':
#     #     height, width = width, height
#     rect = Rectangle((x, y), width, height, edgecolor='red', facecolor='green')
#     ax.add_patch(rect)

# # Set the aspect ratio and display the plot
# ax.set_aspect('equal', 'box')
# plt.show()



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

