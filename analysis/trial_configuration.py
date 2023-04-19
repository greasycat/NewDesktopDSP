from __future__ import annotations
from typing import Tuple, Iterator, Dict, Any, Generator


class TrialConfiguration:
    def __init__(self, row_generator: Generator[str, Any, None]):
        """
        A class for managing trial configurations.

        :param row_generator: An iterator of dictionaries, each containing a trial configuration.
        """
        self.configuration = {}
        for row in row_generator:
            trial_name = row["TrialName"]
            self.configuration[trial_name] = row

    def get_source_destination_pair_by_name(self, name: str) -> Tuple[str, str]:
        """
        Get the source and destination pair for a given trial name.

        :param name: The name of the trial.
        :return: A tuple containing the source and destination.
        :rtype: Tuple[str, str]
        """
        return self.configuration[name]["Source"], self.configuration[name]["Destination"]

    def get_source_destination_pair_by_index(self, index: int) -> Tuple[str, str]:
        """
        Get the source and destination pair for a given trial index.

        :param index: The index of the trial.
        :return: A tuple containing the source and destination.
        :rtype: Tuple[str, str]
        """
        return self.configuration[index]["Source"], self.configuration[index]["Destination"]

    def get_true_angle(self, name: str) -> float:
        """
        Get the true angle for a given trial name.

        :param name: The name of the trial.
        :return: The true angle.
        :rtype: float
        """
        return float(self.configuration[name]["TrueAngle"])
