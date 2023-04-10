import numpy as np


class Grid:
    def __init__(self, origin, map_actual_size, grid_size):
        self.origin = origin
        self.map_actual_size = map_actual_size
        self.grid_size = grid_size
        self.unit_vectors = map_actual_size/grid_size

    def offset_to_origin(self, pos):
        return pos - self.origin

    def get_block_pos(self, pos):
        return np.floor(self.offset_to_origin(pos) / self.unit_vectors) + np.array([1, 1])
