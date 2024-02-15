from src.strippacking2d.problem import StripPacking2D
from src.strippacking2d.solvers.solver_cp_oriented import StripPacking2DCPSolver
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
    items = [{'width': item['Length'], 'height': item["Height"]} for item in asd['Items']]

    return W, items


results = {}

for i in range(6, 13):
    strip_width, rectangles = parse_bkf_benchmark(f"raw_data/2d_strip_packing/BKW/{i}.json")

    # strip_width, rectangles = parse_txt_file("data/2DSTRIPPACKING/zdf2.txt")

    problem = StripPacking2D(benchmark_name="StripPacking2DTest", instance_name="Test01", data={"rectangles": rectangles, "strip_width": strip_width}, solution={}, run_history=[])

    total_height, variables_export, solution = StripPacking2DCPSolver(TimeLimit=3).solve(problem, validate=False, visualize=False, force_execution=True)  # orientations, 

    if variables_export:
        print("Total height:", total_height)
        print("Placement of rectangles:", variables_export['placements'])
        print("orientations:", variables_export['orientations'])
        print("dimensions", rectangles)

        results[f'BKW{i}'] = {"Height": total_height, "Rectangles": [[*rectangle, *placement] for (rectangle, placement) in zip(rectangles, variables_export['placements'])]}

with open("cp_results.json", 'w+', encoding='utf-8') as file:
    json.dump(results, file)
