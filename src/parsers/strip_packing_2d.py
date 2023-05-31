

def print_verbose(print_input, verbose):
    if verbose:
        print(print_input)


def load_strip_packing(instance_path, verbose=False):
    parsed_input = {}

    with open(instance_path) as file:
        # {no_elements}
        no_elements = int(file.readline().strip())
        parsed_input["no_elements"] = no_elements

        # {strip_width}
        strip_width = int(file.readline().strip())
        parsed_input["strip_width"] = strip_width

        parsed_input["elements"] = []
        for i in range(no_elements):
            # {element_width} {element_height}
            line = file.readline()
            _, element_width, element_height = [int(number.strip()) for number in file.readline().split(" ") if len(number) > 0]
            element = {}
            element["element_width"] = element_width
            element["element_height"] = element_height
            
            parsed_input["elements"].append(element)

    return parsed_input

def load_strip_packing_solution(file_path, benchmark_name, instance):
    # parameter = instance.split("_")[0][len(benchmark_name):]
    # instance = instance.split("_")[1].split(".")[0]
    solution = {"feasible": None, "optimum": None, "cpu_time": None}
    
    return solution