from numpy import array_equal

class State:
    def __init__(self, pos, direction, image):
        self.pos = pos
        self.direction = direction
        self.image = str(image)

    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, State):
            return self.pos[0] == other.pos[0] and self.pos[1] == other.pos[1] and self.direction == other.direction and self.image == other.image
        return False

    def __hash__(self) -> int:
        return hash((self.pos[0], self.pos[1], self.direction, self.image))

    def __str__(self) -> str:
        return str(self.pos) + " " + str(self.direction)

