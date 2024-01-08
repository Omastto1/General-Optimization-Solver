# THIS FILE LOADS .BPP FORMAT

def load_1dbinpacking(path, verbose):
    """
    Load the input for 1D Bin Packing in .BPP format 
    Contains the number of rectangles, the bin capacity and the weights of each rectangle
    """
    with open(path, "r") as file:
        line = file.readline()
        no_rectangles = int(line.strip())

        line = file.readline()
        bin_capacity = int(line.strip())

        weights = []
        for _ in range(no_rectangles):
            weight = int(file.readline().strip())
            weights.append(weight)

        parsed_input = {
            "no_rectangles": no_rectangles,
            "bin_capacity": bin_capacity,
            "weights": weights,
        }

        return parsed_input
