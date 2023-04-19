from __future__ import annotations
import numpy as np


class Grid:
    def __init__(self,
                 origin: np.ndarray,
                 map_actual_size: np.ndarray,
                 grid_size: np.ndarray):
        """
        A class representing a grid in the study.

        :param origin: The origin of the grid.
        :param map_actual_size: The actual size of the map.
        :param grid_size: The size of the grid.
        """
        self.origin = origin
        self.map_actual_size = map_actual_size
        self.grid_size = grid_size
        self.unit_vectors = map_actual_size / grid_size

    def offset_to_origin(self, pos: np.ndarray) -> np.ndarray:
        """
        Calculate the offset to the origin.

        :param pos: The position in the grid.
        :return: The offset to the origin.
        :rtype: np.ndarray
        """
        return pos - self.origin

    def get_block_pos(self, pos: np.ndarray) -> np.ndarray:
        """
        Calculate the block position in the grid.

        :param pos: The position in the grid.
        :return: The block position.
        :rtype: np.ndarray
        """
        return np.floor(self.offset_to_origin(pos) / self.unit_vectors) + np.array([1, 1])

    def get_map_pos(self, pos: np.ndarray) -> np.ndarray:
        """
        Calculate the map position in the grid.

        :param pos: The position in the grid.
        :return: The map position.
        :rtype: np.ndarray
        """
        return self.offset_to_origin(pos) / self.unit_vectors
