import numpy as np

from analysis.loader import Loader
from analysis.subject import Subject
from analysis.grid import Grid
from analysis.shortcut_map import ShortcutMap
from analysis.utility import list_sub_one, list_add_one

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from typing import Any, Dict, List, Optional, Tuple

import os


# H11 W11
# TopLeft 0.5, 225
# TopRight 227 225
# BottomLeft 0.5 -1
# BottomRight 227 -1
# for key, value in subjects.movement_sequence.items():


class MovementAnalyzer:
    def __init__(self,
                 loader: Loader,
                 shortcut_map: ShortcutMap,
                 origin=np.array([3, 0]),
                 map_actual_size=np.array([225, 225]),
                 grid_size=np.array([11, 11])):
        self.grid = Grid(origin=origin, map_actual_size=map_actual_size, grid_size=grid_size)
        self.shortcut_map = shortcut_map
        self.trial_configuration = loader.trial_configuration
        self.current_subject = None
        self.bg_1 = loader.image_maze1
        self.subjects = loader.subjects

    def _load_xy(self, subject: Subject, trial_number: int) -> \
            Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Load x, y coordinates for given subject and trial number.

        :param subject: The subject object.
        :param trial_number: The trial number for the subject.
        :return: A tuple containing the original x, y coordinates and the optimized x, y coordinates as numpy arrays.
        :rtype: np.ndarray, np.ndarray, np.ndarray, np.ndarray
        """
        try:
            self.current_subject = subject

            movements = subject.movement_sequence[trial_number]
            trial_name = movements[0].trial_name

            source, destination = self.trial_configuration.get_source_destination_pair_by_name(trial_name)
            _, shortcut = self.shortcut_map.get_shortest_path(source, destination)
            start, end = shortcut[0], shortcut[-1]

            original_movement = [self.grid.get_map_pos(move.get_vector()) for move in movements]

            # convert to 0 indexed
            movements = list_sub_one([self.grid.get_block_pos(move.get_vector()) for move in movements])

            # remove consecutive duplicates
            movements = [movements[i] for i in range(len(movements)) if
                         i == 0 or
                         (not np.array_equal(movements[i], movements[i - 1])) and
                         not self.shortcut_map.check_coord_is_wall(movements[i])]

            fraction_length = len(movements) // 3

            # check if start point is in first third of movements array
            if start in movements[:fraction_length]:
                # remove all movements before start
                movements = movements[movements.index(start):]

            else:
                # add start to movements
                movements = [start] + movements

            # check offset is dot product not equal to 1,
            # then add paths from get_shortest_path_from_two_coords to interpolate
            i = 0
            # is_end_in_movements = False
            # where_is_the_end = 0
            while i < len(movements) - 1:

                # mark if the movement already contains an end
                # if np.array_equal(movements[i + 1], end):
                #     is_end_in_movements = True
                #     where_is_the_end = i + 1

                offset = np.array(movements[i + 1]) - np.array(movements[i])
                if offset.dot(offset) > 1:
                    shortcut = self.shortcut_map.get_shortest_path_from_two_coords(movements[i], movements[i + 1])
                    movements = movements[:i + 1] + shortcut[1:-1] + movements[i + 1:]
                    i += len(shortcut) - 2
                else:
                    i += 1

            # check if end point is in last third of movements array
            if end in movements[-fraction_length:]:
                # remove all the elements after end
                movements = movements[:movements.index(end) + 1]
            else:
                # get the shortest path from last element to end
                shortcut = self.shortcut_map.get_shortest_path_from_two_coords(movements[-1], end)
                if len(shortcut) - 2 < 3:
                    movements = movements + shortcut[1:]

            path = list_add_one(movements)

            arr = np.array(path)
            x = (arr[:, 0]) - 0.5
            y = (arr[:, 1]) - 0.5

            original_movement = np.array(original_movement)
            original_movement = original_movement[np.r_[True, np.any(np.diff(original_movement, axis=0), axis=1)]]

            return original_movement[:, 0], original_movement[:, 1], x, y
        except (IndexError, KeyError):
            return None

    def _draw(self, subject: str, n: int, ox: List[float], oy: List[float], x: List[float], y: List[float],
              bg_file: str = "",
              save_only: bool = False) -> None:
        """
        Draws a path plot for a given subject, trial, and path.

        Args:
            subject (str): The name of the subject.
            n (int): The index of the trial to draw.
            ox (list): The x-coordinates of the original path.
            oy (list): The y-coordinates of the original path.
            x (list): The x-coordinates of the path.
            y (list): The y-coordinates of the path.
            bg_file (str): The filename of the background image to use.
            save_only (bool): If True, save the plot without showing it.

        Returns:
            None
        """

        if bg_file == "":
            bg_file = self.bg_1

        fig, ax = plt.subplots()

        # self.fig, self.ax = plt.subplots()
        ax.step(x, y, color='red', alpha=0.2)
        ax.plot(ox, oy, color='green', alpha=0.2)

        plt.autoscale(False)
        bg = mpimg.imread(bg_file)

        plt.imshow(bg, extent=[0, self.grid.grid_size[0], 0, self.grid.grid_size[1]])

        trial_name = self.subjects[subject].movement_sequence[n][0].trial_name
        source, destination = self.trial_configuration.get_source_destination_pair_by_name(trial_name)
        shortest = self.shortcut_map.get_shortest_path(source, destination)[0]
        estimated_distance = len(x) - 1
        efficiency = estimated_distance / shortest
        if efficiency < 1:
            efficiency = 1
        plt.title(
            f"Subject {subject}\n "
            f"Trial {n - 2}@{trial_name}\n "
            f"From {source} to {destination}\n "
            f"Distance: {estimated_distance} "
            f"Shortest: {shortest} "
            f"Efficiency: {efficiency:.2f}")
        plt.xticks(np.arange(0, self.grid.grid_size[0] + 1, step=1))
        plt.yticks(np.arange(0, self.grid.grid_size[1] + 1, step=1))
        plt.grid()
        fig.set_size_inches(7, 7)

        # create folder if not exist
        if not os.path.exists('path_plot/' + subject):
            os.makedirs('path_plot/' + subject)
        plt.savefig('path_plot/' + subject + '/' + trial_name + '_Trial' + str(n - 2) + '.png', dpi=80)

        if not save_only:
            plt.show()

    def _calculate_efficiency(self, subject: Any, n: int, estimated_distance: int) -> float:
        """
        Calculate efficiency for a given trial based on the estimated distance and shortest path.

        :param subject: The subject object containing movement_sequence and timeout_trials.
        :param n: The trial number.
        :param estimated_distance: The estimated distance of the path.
        :return: The efficiency of the movement path.
        :rtype: float
        """
        trial_name = subject.movement_sequence[n][0].trial_name
        if n in subject.timeout_trials:
            return 2.54
        else:
            source, destination = self.trial_configuration.get_source_destination_pair_by_name(trial_name)
            shortest = self.shortcut_map.get_shortest_path(source, destination)[0]
            efficiency = estimated_distance / shortest
            return max(efficiency, 1)

    def calculate_efficiency_for_one_subject(self, subject_name: str, start: int, end: int) -> Dict[int, float]:
        """
        Calculate efficiency for one subject between the given trial range.

        :param subject_name: The name of the subject.
        :param start: The starting trial number (inclusive).
        :param end: The ending trial number (exclusive).
        :return: A dictionary with trial numbers as keys and efficiency values as values.
        :rtype: Dict[int, float]
        """
        efficiency_dict = {}
        for n in range(start, end):
            _, _, x, _ = self._load_xy(self.subjects[subject_name], n)
            efficiency_dict[n] = self._calculate_efficiency(self.subjects[subject_name], n, len(x) - 1)
        return efficiency_dict

    def calculate_efficiency_for_these_subjects(self, subjects: List[str], start: int = 3, end: int = 23) \
            -> Dict[str, Dict[int, float]]:
        """
        Calculate efficiency for a list of subjects between the given trial range.

        :param subjects: A list of subject names.
        :param start: The starting trial number (inclusive).
        :param end: The ending trial number (exclusive).
        :return: A dictionary with subject names as keys and efficiency dictionaries as values.
        :rtype: Dict[str, Dict[int, float]]
        """
        efficiency_dict = {}
        for subject_name in subjects:
            efficiency_dict[subject_name] = self.calculate_efficiency_for_one_subject(subject_name, start, end)
        return efficiency_dict

    def calculate_efficiency_for_all_subjects(self, start: int = 3, end: int = 23,
                                              excluding: Optional[List[str]] = None) \
            -> dict[Any, dict[int, float | int | Any]]:
        """
        Calculates the efficiency of all subjects in the dataset.

        Args:
            start (int): The index of the first trial to be included in the calculation (inclusive).
            end (int): The index of the last trial to be included in the calculation (exclusive).
            excluding (list): A list of subject names to be excluded from the calculation.

        Returns:
            A dictionary mapping subject names to their efficiency scores.
        """

        if excluding is None:
            excluding = []

        subjects = [subject for subject in self.subjects if subject not in excluding]

        return self.calculate_efficiency_for_these_subjects(subjects, start, end)

    def calculate_failure_for_these_subjects(self, subjects: List[str]) -> Dict[str, int]:
        """
        Calculate failure count for a list of subjects.

        :param subjects: A list of subject names.
        :return: A dictionary with subject names as keys and failure counts as values.
        :rtype: Dict[str, int]
        """
        failure_dict = {}
        for subject in subjects:
            failure_dict[subject] = len(self.subjects[subject].timeout_trials)
        return failure_dict

    def calculate_failure_for_all_subjects(self, excluding: Optional[List[str]] = None) -> Dict[str, int]:
        """
        Calculate failure count for all subjects.

        :param excluding: A list of subject names to be excluded from the calculation.
        :return: A dictionary with subject names as keys and failure counts as values.
        :rtype: Dict[str, int]
        """
        if excluding is None:
            excluding = []

        subjects = [subject for subject in self.subjects if subject not in excluding]
        return self.calculate_failure_for_these_subjects(subjects)

    def plot_for_these_subjects(self, subjects: List[str], start: int = 3, end: int = 23, save_only: bool = False):
        """
        Plot movement paths for a list of subjects between the given trial range.

        :param subjects: A list of subject names.
        :param start: The starting trial number (inclusive).
        :param end: The ending trial number (exclusive).
        :param save_only: If True, the plot will only be saved and not displayed.
        """
        for subject in subjects:
            for n in range(start, end):
                ox, oy, x, y = self._load_xy(self.subjects[subject], n)
                self._draw(subject=subject, n=n, ox=ox, oy=oy, x=x, y=y, save_only=save_only)

    def plots_for_all_subjects(self, start=3, end=23, excluding=None, save_only=False):
        """
        Saves plots for all subjects in the dataset.

        Args:
            start (int): The index of the first trial to be plotted (inclusive).
            end (int): The index of the last trial to be plotted (exclusive).
            excluding (list): A list of subject names to be excluded from the plots.
            save_only (bool): If True, save the plots without showing them.

        Returns:
            None
        """
        if excluding is None:
            excluding = []

        subjects = [subject for subject in self.subjects if subject not in excluding]

        self.plot_for_these_subjects(subjects, start, end, save_only)
