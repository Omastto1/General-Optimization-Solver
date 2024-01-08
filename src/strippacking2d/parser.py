import json
import pandas as pd


def print_verbose(print_input, verbose):
    if verbose:
        print(print_input)


def load_bkw_benchmark(file_path, verbose=False):
    asd = json.loads(open(file_path, 'r', encoding='utf-8').read())

    strip_width = asd['Objects'][0]['Length']
    items = [{"width": item['Length'], 'height': item["Height"]}
             for item in asd['Items'] for _ in range(item['Demand'])]

    data = {"strip_width": strip_width, "rectangles": items}

    return data


def load_bkw_benchmark_solution(file_path, instance_name: str, verbose=False):
    solutions = pd.read_csv(file_path, sep=';')
    obj_value = solutions[solutions['instance_id'] == int(
        instance_name)]['obj_value'].astype(int).values[0]

    # cast obj_value to int, np.int64 is not json serializable
    solution = {"feasible": True, "optimum": int(obj_value)}

    return solution


def load_strip_packing(instance_path, verbose=False):
    parsed_input = {}

    with open(instance_path, 'r', encoding='utf-8') as file:
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
            _, element_width, element_height = [
                int(number.strip()) for number in line.split() if len(number) > 0]
            element = {}
            element["width"] = element_width
            element["height"] = element_height

            parsed_input["rectangles"].append(element)

    return parsed_input


def load_strip_packing_solution(file_path, instance):
    solution = {"feasible": None, "optimum": None,
                "cpu_time": None, "bounds": None}

    return solution
