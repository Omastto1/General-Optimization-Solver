import numpy as np


def fitness_func(instance, x, out):
    """
    Place rectangles one after each other by one into levels, 
    where height of level is defined by the height of the largest
    rectangle

    Return total height as the sum of levels heights
    """
    order = np.argsort(x)
    total_height = 0
    current_width = 0
    current_height = 0
    rectangles = [None] * instance.no_elements
    for i in order:
        if current_width + instance.rectangles[i]['width'] > instance.strip_width:
            total_height += current_height
            current_width = 0
            current_height = 0

        rectangles[i] = (current_width, total_height)
        
        current_width += instance.rectangles[i]['width']
        current_height = max(current_height, instance.rectangles[i]['height'])
    total_height += current_height

    out["F"] = total_height
    out["rectangles"] = rectangles

    return out