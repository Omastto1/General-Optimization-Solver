import copy
import matplotlib.pyplot as plt

from matplotlib.patches import Rectangle as PltRectangle

from ga_fitness_functions.strip_packing_2d.best_fit_ga.rectangle_utils import rotate_rectangles, sort_rectangles, find_highest_rectangle
from ga_fitness_functions.strip_packing_2d.best_fit_ga.rectangle_utils import sort_rectangles_penalty

from ga_fitness_functions.strip_packing_2d.best_fit_ga.rectangle import RectanglePenalty


def squeeky_wheel_optimization_wrapper(instance, x, out):
    # print("running \n")
    # Calculate the total height based on the order in x
    rectangles = [RectanglePenalty(rectangle['width'], rectangle['height'], index, round(penalty)) for index, (rectangle, penalty) in enumerate(zip(instance.rectangles, x))]

    skyline, rectangles = squeeky_wheel_optimization_ga(rectangles, instance.strip_width)

    total_height = max(skyline)
    rectangles.sort(key=lambda x: x.index, reverse=False)

    # rectangles = [(rectangle.x_placement, rectangle.y_placement) for rectangle in rectangles]

    out["F"] = total_height
    out["rectangles"] = rectangles

    return out

fitness_func = squeeky_wheel_optimization_wrapper


def visualize(rectangles, strip_width, total_height):
    fig, ax = plt.subplots()
    ax.set_xlim([0, strip_width])
    ax.set_ylim([0, total_height])

    # Draw the large rectangle
    large_rect = PltRectangle((0, 0), strip_width, total_height, edgecolor='black', facecolor='none')
    ax.add_patch(large_rect)

    # Draw the small rectangles within the large rectangle
    for index, rectangle in enumerate(rectangles):
        # if index < 10:
            # print(rectangle)
        rect = PltRectangle((rectangle.x_placement, rectangle.y_placement), rectangle.width, rectangle.height, edgecolor='red', facecolor='green')
        ax.add_patch(rect)

    # Set the aspect ratio and display the plot
    ax.set_aspect('equal', 'box')
    plt.show()


""" GAP FINDER """

def find_lowest_gap(skyline):
    """
    Find the starting index and width of the lowest gap in the skyline.

    Parameters:
        skyline (list of int): The skyline array.

    Returns:
        int, int: Starting index and width of the gap.
    """

    gap_start = -1  # Initialize the starting index of the gap
    gap_width = 0   # Initialize the width of the gap

    min_height = float('inf')  # Initialize the smallest height as infinity

    for i, height in enumerate(skyline):
        # Update the minimum height if a smaller height is found
        if height < min_height:
            min_height = height
            gap_start = i  # Update the gap start to the current index
            gap_width = 1  # Start counting the gap width
            extending_gap = True

        # Increase the gap width if a height is equal to the smallest height
        elif height == min_height and extending_gap:
            gap_width += 1

        # Exit the loop when exited smallest height section
        elif height > min_height and gap_start != -1:
            extending_gap = False

    if gap_start == -1:
        raise ValueError("No gap found in the skyline.")

    return gap_start, gap_width, min_height

""" END GAP FINDER """

""" RECTANGLE PLACEMENT """

def find_best_fitting_rectangle(rectangles, gap_width):
    """
    Find the best-fitting rectangle.

    Parameters:
        rectangles (list of Rectangle): The list of rectangles.
        gap_width (int): The width of the gap.

    Returns:
        Rectangle: The rectangle to be placed or None if no suitable rectangle is found.
    """
    best_fit = None
    best_fit_dim = 0

    if gap_width == 50:
        print("asd")

    for rect in rectangles:
        if rect.is_placed:
            continue

        # Check if the rectangle can fill the gap in either orientation
        if rect.can_fill_gap(gap_width):

            # TODO: THERE MAY BE MULTIPLE RECTANGLE WITH SAME WIDTH BUT THE OTHER ONE MAY BE HIGHER
            # Check if the rectangle fits exactly, and if so, place it immediately
            if rect.width == gap_width or rect.height == gap_width:
                rect.is_placed = True
                # Make sure to rotate the rectangle if its height matches the gap width
                if rect.height == gap_width:
                    rect.rotate()
                return rect

            # If either width or height is a better fit than the current best, update the best fit
            if rect.width > best_fit_dim and rect.width <= gap_width:
                best_fit = rect
                best_fit_dim = rect.width
            elif rect.height > best_fit_dim and rect.height <= gap_width:
                best_fit = rect
                best_fit_dim = rect.height
                rect.rotate()

    return best_fit


def place_rectangle_left(rectangle, gap_start, skyline):
    rectangle.place(gap_start, skyline[gap_start])

def place_rectangle_right(rectangle, gap_start, gap_width, skyline):
    rectangle.place(gap_start + (gap_width - rectangle.width), skyline[gap_start])


def place_rectangle(rectangle, gap_start, gap_width, policy, skyline):
    """policies = ["Leftmost", "Tallest Neighbour", "Smallest Neighbour"]
    """
    # TODO: UPDATE FOR DIFFERENT POLICIES

    if policy == "Leftmost":
        place_rectangle_left(rectangle, gap_start, skyline)
    else:
        left_neighbor_height = (skyline[gap_start - 1:] + [float('inf')])[0]
        right_neighbor_height = (skyline[gap_start + rectangle.width:] + [float('inf')])[0]

        if policy == "Tallest Neighbour":
            if left_neighbor_height >= right_neighbor_height:
                place_rectangle_left(rectangle, gap_start, skyline)
            else:
                place_rectangle_right(rectangle, gap_start, gap_width, skyline)

        elif policy == "Smallest Neighbour":
            if left_neighbor_height <= right_neighbor_height:
                place_rectangle_left(rectangle, gap_start, skyline)
            else:
                place_rectangle_right(rectangle, gap_start, gap_width, skyline)

        else:
            raise ValueError("Invalid policy specified.")

""" END RECTANGLE PLACEMENT """

""" SKYLINE UPDATE """

def update_skyline_with_rectangle(skyline, rectangle):
    """
    Update the skyline array based on the placed rectangle.

    Parameters:
        skyline (list of int): The skyline array.
        rectangle (Rectangle): The rectangle to be placed.

    Returns:
        None: The skyline array is modified in place.
    """
    if rectangle.x_placement < 0 or rectangle.x_placement + rectangle.width > len(skyline):
        raise ValueError(
            "The rectangle does not fit into the skyline at the specified x position.")

    # Update the skyline array by adding the rectangle's height
    # to each position from rectangle.x_placement to rectangle.x_placement + rectangle.width
    for i in range(rectangle.x_placement, rectangle.x_placement + rectangle.width):
        skyline[i] += rectangle.height

def update_skyline_with_waste(skyline, gap_start, gap_width):
    """
    Update the skyline by filling a gap to the lowest neighboring height.

    Parameters:
        skyline (list of int): The skyline array.
        gap_start (int): The start index of the gap in the skyline array.
        gap_end (int): The width of the gap.

    Returns:
        None: The skyline array is modified in place.
    """
    
    # Validate the inputs
    if gap_start < 0 or gap_start + gap_width > len(skyline):
        raise ValueError("Invalid gap specification: out of skyline bounds.")
    
    # Find the heights to the left and right of the gap, if they exist
    left_height = skyline[gap_start - 1] if gap_start > 0 else float('inf')
    right_height = skyline[gap_start + gap_width] if gap_start + gap_width < len(skyline) else float('inf')
    
    # Determine the height to which the gap should be raised: the lower of the two neighboring heights
    fill_height = min(left_height, right_height)
    if fill_height == float('inf'):
        raise ValueError("Invalid gap specification: no neighboring heights found. Check if gap_width <= all rectangles' width")
    
    # Update the skyline by raising the gap to fill_height
    for i in range(gap_start, gap_start + gap_width):
        skyline[i] = fill_height


def reduce_skyline(skyline, rectangle):
    """
    Decrease the skyline by the height of the rectangle from its 
    x_placement to the width of the rectangle.

    Parameters:
        rectangle (Rectangle): The rectangle to be removed.
        skyline (list): A list representing the skyline heights.
    """
    for i in range(rectangle.x_placement, rectangle.x_placement + rectangle.width):
        # Ensure that the skyline does not go below zero.
        skyline[i] = skyline[i] - rectangle.height


def reset_skyline(strip_width):
    return [0] * strip_width
        
""" END SKYLINE UPDATE """

def optimize_solution(rectangles, skyline, policy, strip_width):
    """
    Optimize the placement of shapes in a skyline.

    Parameters:
        shapes (list): A list of shapes to be placed.
        skyline (list): A list representing the current skyline.

    Returns:
        list: The optimized skyline.
    """
    skyline_old = copy.deepcopy(skyline)
    rectangles_old = copy.deepcopy(rectangles)
    old_height = max(skyline_old)

    while True:
        highest_rectangle = find_highest_rectangle(rectangles, skyline)

        # Check termination condition based on shape dimensions
        if highest_rectangle.width >= highest_rectangle.height:
            break
        
        reduce_skyline(skyline, highest_rectangle)
        highest_rectangle.deassign()
        highest_rectangle.rotate()

        if highest_rectangle.width > strip_width:
            break

        while not highest_rectangle.is_placed:
            gap_start, gap_width = find_lowest_gap(skyline)
            
            if highest_rectangle.width <= gap_width:
                place_rectangle(highest_rectangle, gap_start, gap_width, policy, skyline)
                update_skyline_with_rectangle(skyline, highest_rectangle)

                if max(skyline) < old_height:
                    skyline_old = copy.deepcopy(skyline)
                    rectangles_old = copy.deepcopy(rectangles)
                    break
            else:
                update_skyline_with_waste(skyline, gap_start, gap_width)

        # Check termination condition based on packing improvement
        if max(skyline) >= old_height:
            break
    
    return skyline_old, rectangles_old

def pack_rectangles(rectangles, skyline, policy, strip_width, best_fit_selector, enable_optimization, verbose=False):
    is_missing_rectangle_placement = True

    while is_missing_rectangle_placement:
        gap_start, gap_width, min_height = find_lowest_gap(skyline)
        # if min_height > 350:
        #     print("asd")

        # if gap_start > 350:
        #     print("asd")

        best_fitting_rectangle = best_fit_selector(
            rectangles, gap_width)
        
        if best_fitting_rectangle is None and gap_width == strip_width:
            for rectangle in rectangles:
                if not rectangle.is_placed:
                    rectangle.width, rectangle.height = rectangle.height, rectangle.width
            
            best_fitting_rectangle = best_fit_selector(
            rectangles, gap_width)
        
        if best_fitting_rectangle is not None:
            place_rectangle(best_fitting_rectangle,
                            gap_start, gap_width, policy, skyline)
            update_skyline_with_rectangle(skyline, best_fitting_rectangle)
            
            is_missing_rectangle_placement = any(map(lambda rectangle: not rectangle.is_placed, rectangles))
        else:
            update_skyline_with_waste(skyline, gap_start, gap_width)
    
    if enable_optimization:
        skyline, rectangles = optimize_solution(rectangles, skyline, policy, strip_width)

    if verbose:
        print(f"{policy} policy got {max(skyline)} height")
    # visualize(rectangles, strip_width, max(skyline))

    return skyline, rectangles


def best_fit_heuristics(input_rectangles, strip_width):
    rotate_rectangles(input_rectangles)
    sort_rectangles(input_rectangles)

    policies = ["Leftmost", "Tallest Neighbour", "Smallest Neighbour"]
    best_policy = None
    best_skyline = [float('inf')]
    best_rectangles = None

    for policy in policies:
        skyline = reset_skyline(strip_width)
        rectangles = copy.deepcopy(input_rectangles)

        skyline, rectangles = pack_rectangles(rectangles, skyline, policy, strip_width, find_best_fitting_rectangle, True)
        
        if max(skyline) < max(best_skyline):
            print(f"Setting new best skyline with {policy} policy")
            print("height", max(skyline))

            best_skyline = copy.deepcopy(skyline)
            best_rectangles = copy.deepcopy(rectangles)
            best_policy = policy

    return best_skyline, best_rectangles, best_policy



def find_best_fit_constrained_by_penalty(rectangles, gap_width):
    width_fitting = list(filter(lambda x: x.width <= gap_width and not x.is_placed, rectangles))
    if len(width_fitting) == 0:
        return None
    
    max_penalty = max(map(lambda x: x.penalty, width_fitting))
    max_penalty_rectangles = list(filter(lambda x: x.penalty == max_penalty, width_fitting))

    if len(max_penalty_rectangles) > 1:
        max_width = max(map(lambda x: x.width, max_penalty_rectangles))
        max_width_rectangles = list(filter(lambda x: x.width == max_width, max_penalty_rectangles))

        if len(max_width_rectangles) > 1:
            max_height = max(map(lambda x: x.height, max_width_rectangles))
            max_height_rectangles = list(filter(lambda x: x.height == max_height, max_width_rectangles))
            return rectangles[rectangles.index(max_height_rectangles[0])]
        return rectangles[rectangles.index(max_width_rectangles[0])]
    
    return rectangles[rectangles.index(max_penalty_rectangles[0])]

def squeeky_wheel_optimization_ga(input_rectangles, strip_width):
    rotate_rectangles(input_rectangles)
    sort_rectangles_penalty(input_rectangles)

    policy = "Tallest Neighbour"
    skyline = reset_skyline(strip_width)
    rectangles = copy.deepcopy(input_rectangles)

    skyline, rectangles = pack_rectangles(rectangles, skyline, policy, strip_width, find_best_fit_constrained_by_penalty, False)

    return skyline, rectangles