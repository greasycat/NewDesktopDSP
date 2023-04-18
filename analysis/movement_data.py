from typing import Optional
import numpy as np


class MovementData:
    def __init__(self, trial_name: str, trial_number: int, trial_time: float, x: float, y: float, rotation: float):
        """
        Initialize a MovementData object with trial information and coordinates.

        :param trial_name: The name of the trial.
        :param trial_number: The trial number.
        :param trial_time: The trial time.
        :param x: The x coordinate of the movement.
        :param y: The y coordinate of the movement.
        :param rotation: The rotation value.
        """
        self.y = y
        self.x = x
        self.trial_time = trial_time
        self.trial_name = trial_name
        self.trial_number = trial_number
        self.rotation = rotation

    def get_vector(self) -> np.ndarray:
        """
        Get the x, y coordinate vector of the movement.

        :return: A numpy array containing the x, y coordinates.
        :rtype: np.ndarray
        """
        return np.array([self.x, self.y])

    def get_rotation(self) -> float:
        return self.rotation

    @staticmethod
    def from_str(s: str = "") -> Optional["MovementData"]:
        """
        Create a MovementData object from a string representation.

        :param s: The string representation of the MovementData object.
        :return: A MovementData object if the string can be parsed, otherwise None.
        :rtype: Optional[MovementData]
        """
        elements = [element.strip() for element in s.split(",")]
        if len(elements) < 6:
            return None
        return MovementData(elements[0], int(elements[1]), float(elements[2]), float(elements[3]), float(elements[4]),
                            float(elements[5]))
