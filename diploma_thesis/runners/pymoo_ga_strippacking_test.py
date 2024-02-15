import numpy as np
from pymoo.core.problem import Problem
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.operators.crossover.pntx import TwoPointCrossover


from pymoo.problems import get_problem
from pymoo.optimize import minimize
from pymoo.termination import get_termination

problem = get_problem("g1")




# Sample rectangles: (width, height)
rectangles = np.array([
    [3, 4],
    [5, 6],
    [2, 8],
    [4, 5]
])
strip_width = 5

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

strip_width, rectangles = parse_txt_file("data/2DSTRIPPACKING/zdf2.txt")
rectangles = np.array(rectangles)

class StripPackingProblem(ElementwiseProblem):

    def __init__(self, rectangles, strip_width):
        super().__init__(n_var=len(rectangles),
                         n_obj=1,
                         n_constr=0,
                         xl=0,
                         xu=len(rectangles)-1,
                         elementwise_evaluation=True)

        self.rectangles = rectangles
        self.strip_width = strip_width

    def _evaluate(self, x, out, *args, **kwargs):
        # Calculate the total height based on the order in x
        order = np.argsort(x)
        total_height = 0
        current_width = 0
        current_height = 0
        for i in order:
            if current_width + self.rectangles[i, 0] > self.strip_width:
                total_height += current_height
                current_width = 0
                current_height = 0
            
            current_width += self.rectangles[i, 0]
            current_height = max(current_height, self.rectangles[i, 1])
        total_height += current_height

        out["F"] = total_height

problem = StripPackingProblem(rectangles, strip_width)
termination = get_termination("n_eval", 300)

algorithm = GA(
    pop_size=200,
    sampling=FloatRandomSampling(),
    crossover=TwoPointCrossover(prob=1.0),
    mutation=PolynomialMutation(),
    eliminate_duplicates=True
)

res = minimize(problem, algorithm, ("n_gen", 200), verbose=True)

X = res.X
F = res.F

print("Best solution found: %s" % X)
print("All solutions:")
for x, f in zip(X, F):
    print("- x: %s, f: %s" % (x, f))


order = np.argsort(X)
total_height = 0
current_width = 0
current_height = 0
current_level = 0
placements = [[] for i in range(len(rectangles))]
for i in order:
    if current_width + rectangles[i, 0] > strip_width:
        total_height += current_height
        current_level = total_height
        current_width = 0
        current_height = 0
    
    placements[i] = (current_width, current_level)
    current_width += rectangles[i, 0]
    current_height = max(current_height, rectangles[i, 1])
total_height += current_height

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

total_height = np.min(F)

rectangles_visu = []
for i, rectangle in enumerate(problem.rectangles):
    x, y = placements[i]
    width, height = rectangle
    rectangles_visu.append((x, y, width, height))

# Create a figure and axis for plotting
fig, ax = plt.subplots()
ax.set_xlim([0, problem.strip_width])
ax.set_ylim([0, total_height])

# Draw the large rectangle
large_rect = Rectangle((0, 0), problem.strip_width, total_height, edgecolor='black', facecolor='none')
ax.add_patch(large_rect)

# Draw the small rectangles within the large rectangle
for i, (x, y, width, height) in enumerate(rectangles_visu):
    print(x, y, width, height)
    # if orientations[i] == 'rotated':
    #     height, width = width, height
    rect = Rectangle((x, y), width, height, edgecolor='red', facecolor='green')
    ax.add_patch(rect)

# Set the aspect ratio and display the plot
ax.set_aspect('equal', 'box')
plt.show()