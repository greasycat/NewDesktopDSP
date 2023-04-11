import numpy as np

from analysis.loader import Loader

from math import cos, sin
from typing import Dict, Optional, List


class RotationAnalyzer:
    def __init__(self, loader: Loader):
        """
        A class for analyzing rotation data.

        :param loader: The Loader object containing the dataset.
        """
        self.loader = loader

    def calculate_estimation_error(self, subject_name: str, trial_number: int) -> Optional[float]:
        """
        Calculate the estimation error for a given subject and trial number.

        :param subject_name: The name of the subject.
        :param trial_number: The trial number.
        :return: The estimation error.
        :rtype: float
        """
        try:
            rotation_sequence = self.loader.subjects[subject_name].rotation_sequence
            trial_name = rotation_sequence[trial_number].trial_name
            true_angle = self.loader.trial_configuration.get_true_angle(trial_name)
            estimation_angle = rotation_sequence[trial_number].rotation

            rad_true_angle = np.deg2rad(true_angle)
            rad_estimation_angle = np.deg2rad(estimation_angle)

            v1 = np.array([cos(rad_true_angle), sin(rad_true_angle)])
            v2 = np.array([cos(rad_estimation_angle), sin(rad_estimation_angle)])
            v1_u = v1 / np.linalg.norm(v1)
            v2_u = v2 / np.linalg.norm(v2)

            rad_diff = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
            diff = np.rad2deg(rad_diff)
            return diff
        except (KeyError, IndexError):
            print(f"Error: {subject_name} or {trial_number} data files corrupted. Skipping...")
            return None

    def calculate_estimation_error_for_one(self,
                                           subject_name: str,
                                           start: int = 3,
                                           end: int = 23) -> Dict[int, float]:
        """
        Calculate estimation errors for a specific subject in the given range of trials.

        :param subject_name: The name of the subject.
        :param start: The index of the first trial (inclusive).
        :param end: The index of the last trial (exclusive).
        :return: A dictionary of estimation errors for each trial.
        :rtype: Dict[int, float]
        """
        errors = {}
        for trial_number in range(start, end):
            errors[trial_number] = self.calculate_estimation_error(subject_name, trial_number)
        return errors

    def calculate_all_estimation_error_for_all(self,
                                               start: int,
                                               end: int,
                                               excluding: Optional[List[str]] = None) -> Dict[str, Dict[int, float]]:
        """
        Calculate estimation errors for all subjects in the dataset.

        :param start: The index of the first trial (inclusive).
        :param end: The index of the last trial (exclusive).
        :param excluding: A list of subject names to exclude from the calculation.
        :return: A dictionary of estimation errors for each subject and trial.
        :rtype: Dict[str, Dict[int, float]]
        """
        if excluding is None:
            excluding = []
        errors = {}
        for subject in self.loader.subjects:
            if subject in excluding:
                continue
            errors[subject] = self.calculate_estimation_error_for_one(subject, start, end)
        return errors
