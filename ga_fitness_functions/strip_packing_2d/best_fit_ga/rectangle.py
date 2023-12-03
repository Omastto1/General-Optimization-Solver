class Rectangle:
    def __init__(self, width, height, index):
        self.width = width
        self.height = height
        self.index = index

        self.is_placed = False  # Flag indicating whether the rectangle is placed
        self.x_placement = None
        self.y_placement = None

    def rotate(self):
        self.width, self.height = self.height, self.width

    def can_fill_gap(self, gap_width):
        # Check both orientations (width and height) to see if the rectangle can fill the gap
        return self.width <= gap_width or self.height <= gap_width

    def __str__(self):
        """
        Return a string representation of the Rectangle instance.

        Returns:
            str: A string representation of the rectangle.
        """
        # Basic rectangle properties
        rect_str = f"Rectangle(width={self.width}, height={self.height})"

        # Additional placement details if the rectangle has been placed
        if self.is_placed:
            rect_str += f", placed at ({self.x_placement}, {self.y_placement})"
        else:
            rect_str += ", not placed"

        return rect_str

    def get_top(self):
        if not self.is_placed:
            return None

        return self.y_placement + self.height

    def deassign(self):
        """
        Mark the rectangle as not placed and remove its placement coordinates.
        """
        self.is_placed = False
        self.x_placement = None
        self.y_placement = None

    def place(self, x, y):
        self.is_placed = True
        self.x_placement = x
        self.y_placement = y


class RectanglePenalty(Rectangle):
    def __init__(self, width, height, index, penalty):
        super().__init__(width, height, index)
        self.penalty = penalty