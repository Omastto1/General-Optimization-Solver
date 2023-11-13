

def print_verbose(print_input, verbose):
    if verbose:
        print(print_input)


def load_strip_packing(instance_path, verbose=False):
    parsed_input = {}

    with open(instance_path) as file:
        # {no_elements}
        no_rectangles = int(file.readline().strip())
        parsed_input["no_rectangles"] = no_rectangles

        # {strip_width}
        strip_width = int(file.readline().strip())
        parsed_input["strip_width"] = strip_width

        parsed_input["rectangles"] = []
        for i in range(no_rectangles):
            # {element_width} {element_height}
            line = file.readline()
            _, element_width, element_height = [int(number.strip()) for number in line.split() if len(number) > 0]
            element = {}
            element["width"] = element_width
            element["height"] = element_height
            
            parsed_input["rectangles"].append(element)

    return parsed_input

def load_strip_packing_solution(file_path, instance):
    # parameter = instance.split("_")[0][len(benchmark_name):]
    # instance = instance.split("_")[1].split(".")[0]
    solution = {"feasible": None, "optimum": None, "cpu_time": None, "bounds": None}
    
    return solution