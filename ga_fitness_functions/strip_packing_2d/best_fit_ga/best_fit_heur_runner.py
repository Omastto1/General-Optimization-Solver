from ga_fitness_functions.strip_packing_2d.best_fit_ga.rectangle import Rectangle
from ga_fitness_functions.strip_packing_2d.best_fit_ga.best_fit import best_fit_heuristics, visualize


# Example usage
rectangles = [
    Rectangle(1, 2), Rectangle(5, 3), Rectangle(7, 3),
    Rectangle(5, 2)
]

rectangles = [
    Rectangle(4, 3), Rectangle(3, 3), Rectangle(3, 1), Rectangle(2, 4), Rectangle(2, 2), Rectangle(1, 3), Rectangle(1, 7)
]

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
        items.append(Rectangle(width, height))

    return W, items


strip_width, rectangles = parse_txt_file("data/2DSTRIPPACKING/zdf2.txt")
strip_width = 130


# Example of usage
skyline, rectangles, policy = best_fit_heuristics(rectangles, strip_width)

total_height = max(skyline)

for rectangle in rectangles:
    print(rectangle)

print("asd")


print("Total height:",total_height)

visualize(rectangles, strip_width, total_height)

print("asd")