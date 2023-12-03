def rotate_rectangles(rectangles):
    for rectangle in rectangles:
        if rectangle.height > rectangle.width:
            rectangle.width, rectangle.height = rectangle.height, rectangle.width


def sort_rectangles(rectangles):
    rectangles.sort(key=lambda x: (x.width, x.height), reverse=True)

def sort_rectangles_penalty(rectangles):
    rectangles.sort(key=lambda x: (x.penalty, x.width, x.height), reverse=True)


def find_highest_rectangle(rectangles, skyline):
    """
    Find the highest rectangle.

    Parameters:
        rectangles (list): A list of placed rectangles.
        skyline (list): A list representing the skyline heights.

    Returns:
        Rectangle: The highest rectangle.
    """
    highest_rectangle = None
    highest_top = -1
    
    for rectangle in rectangles:
        if rectangle.is_placed:
            rect_top = rectangle.get_top()
            if rect_top > highest_top:
                highest_rectangle = rectangle
                highest_top = rect_top
    
    return highest_rectangle