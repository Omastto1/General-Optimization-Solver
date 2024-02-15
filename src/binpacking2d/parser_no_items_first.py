# THIS FILE LOADS .BPP FORMAT FOR 2D BINPACKING
###
# {no items}
# {bin_width} {bin_height}
# {items...}
 
def load_2dbinpacking_no_items_first(path, verbose):
    """
    Load the input for 2D Bin Packing in .BPP format 
    Contains the the number of rectangles, bin size and the dimensions of each rectangle
    """
    with open(path, "r") as file:
        line = file.readline()        
        no_items = int(line.strip())
        
        line = file.readline()        
        bin_size = [int(number) for number in line.strip().split(" ")]

        items_sizes = []
        for _ in range(no_items):
            line = file.readline()
            item_size = [int(number) for number in line.strip().split(" ")]

            items_sizes.append(item_size)

        parsed_input = {
            "no_items": no_items,
            "bin_size": bin_size,
            "items_sizes": items_sizes,
        }
        
    return parsed_input
