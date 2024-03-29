from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

from analysis.loader import Loader

from math import cos, sin, fmod
from typing import Dict, Optional, List
import csv


def angle_difference(angle1: float, angle2: float) -> float:
    return abs(angle1 - angle2)


class RotationAnalyzer:
    def __init__(self, loader: Loader):
        """
        A class for analyzing rotation data.

        :param loader: The Loader object containing the dataset.
        """
        self.loader = loader

    def plot_estimation_error(self, subject_name: str, trial_number: int):
        """
        Plot the estimation error for a given subject and trial number.

        :param subject_name: The name of the subject.
        :param trial_number: The trial number.
        """
        rotation_sequence = self.loader.subjects[subject_name].rotation_sequence
        trial_name = rotation_sequence[trial_number].trial_name
        true_angle = self.loader.trial_configuration.get_true_angle(trial_name) / 180 * np.pi
        estimation_angle = rotation_sequence[trial_number].rotation / 180 * np.pi

        x = np.linspace(-5, 5, 100)
        y1 = np.tan(true_angle + np.pi / 2) * x
        y2 = np.tan(estimation_angle + np.pi / 2) * x

        r = 4
        a = np.linspace(0, 2 * np.pi, 100)
        xs = r * np.cos(a)
        ys = r * np.sin(a)

        fig, ax = plt.subplots()

        ax.plot(xs, ys, label='Circle')
        ax.plot(x, np.tan(np.pi / 2) * x, label='90', color='black')
        ax.plot(x, np.tan(np.pi) * x, label='180', color='black')

        ax.plot(x, y1, label='True Angle')
        ax.plot(x, y2, label='Estimation Angle')
        # set both limits from -5 to 5
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_title(f"Estimation Error for {subject_name} Trial {trial_name}"
                     f"\nTrue Angle: {true_angle * 180 / np.pi} Estimation Angle: {estimation_angle * 180 / np.pi}"
                     f"\n{angle_difference(true_angle * 180 / np.pi, estimation_angle * 180 / np.pi)}")
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.legend()
        plt.gca().set_aspect('equal')
        plt.show()
        pass

    def plot_estimation_error_for_one_subject(self, subject_name: str, start: int = 3, end: int = 23):
        """
        Plot the estimation error for a given subject and trial number.
        :param subject_name: The name of the subject.
        :param start: The start trial number.
        :param end: The end trial number.
        """

        for trial_number in range(start, end):
            self.plot_estimation_error(subject_name, trial_number)

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

            # rad_true_angle = np.deg2rad(true_angle)
            # rad_estimation_angle = np.deg2rad(estimation_angle)
            #
            # v1 = np.array([cos(rad_true_angle), sin(rad_true_angle)])
            # v2 = np.array([cos(rad_estimation_angle), sin(rad_estimation_angle)])
            # v1_u = v1 / np.linalg.norm(v1)
            # v2_u = v2 / np.linalg.norm(v2)
            #
            # rad_diff = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
            # diff = np.rad2deg(rad_diff)
            # complementary_diff = 180 - diff
            # return min(diff, complementary_diff)
            diff = angle_difference(true_angle, estimation_angle)
            return abs(min(diff, 360 - diff))
        except (KeyError, IndexError):
            print(f"Error: {subject_name} or {trial_number} data files corrupted. Skipping...")
            return None

    def calculate_estimation_error_for_one_subject(self,
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

    def calculate_estimation_error_for_all_subjects(self,
                                                    start: int = 3,
                                                    end: int = 23,
                                                    excluding: Optional[List[str]] = None) -> \
            Dict[str, Dict[int, float]]:
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
            errors[subject] = self.calculate_estimation_error_for_one_subject(subject, start, end)
        return errors

    @staticmethod
    def export(errors, file_name: str = 'estimation_errors.csv'):
        """
        Export the estimation errors to a csv file.
        :param errors: The estimation errors.
        :param file_name: The name of the csv file.
        """
        with open(file_name, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(['Subject', 'Trial', 'Estimation Error'])
            for subject in errors:
                for trial in errors[subject]:
                    writer.writerow([subject, trial, errors[subject][trial]])

    def export_estimation_error_for_all_subjects(self, start: int = 3, end: int = 23,
                                                 file_name: str = 'estimation_errors.csv'):
        """
        Export the estimation errors for all subjects in the dataset.
        :param start: The index of the first trial (inclusive).
        :param end: The index of the last trial (exclusive).
        :param file_name: The name of the csv file.
        """
        errors = self.calculate_estimation_error_for_all_subjects(start, end)
        self.export(errors, file_name)
