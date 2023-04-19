from __future__ import annotations
from typing import Dict, Union


class RotationData:
    def __init__(self,
                 trial_name: str,
                 trial_number: int,
                 trial_time: float,
                 rotation: float):
        """
        A class representing rotation data.

        :param trial_name: The name of the trial.
        :param trial_number: The trial number.
        :param trial_time: The time of the trial.
        :param rotation: The rotation angle.
        """
        self.trial_time = trial_time
        self.trial_name = trial_name
        self.trial_number = trial_number
        self.rotation = rotation

    @staticmethod
    def from_dict(d: Dict[str, Union[str, int, float]]) -> "RotationData":
        """
        Create a RotationData object from a dictionary.

        :param d: The dictionary containing the rotation data.
        :return: A RotationData object.
        :rtype: RotationData
        """
        return RotationData(d["TrialName"], d["Trial"], float(d["Time"]), float(d["Angle"]))
