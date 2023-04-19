from __future__ import annotations
import numpy as np

from analysis.loader import Loader
from analysis.movement_data import MovementData
from analysis.subject import Subject
from analysis.grid import Grid
from analysis.shortcut_map import ShortcutMap
from analysis.utility import list_sub_one, list_add_one, tuple_sub_one, tuple_add_one, tuple_add

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
                 learning_map: ShortcutMap,
                 origin=np.array([3, 0]),
                 map_actual_size=np.array([225, 225]),
                 grid_size=np.array([11, 11])):
        self.grid = Grid(origin=origin, map_actual_size=map_actual_size, grid_size=grid_size)
        self.shortcut_map = shortcut_map
        self.learning_map = learning_map
        self.trial_configuration = loader.trial_configuration
        self.current_subject = None
        self.bg_1 = loader.image_maze1
        self.subjects = loader.subjects
        self.cache: Dict[str, Any] = {}

    def get_source_destination(self, subject_name, trial_number):
        movements = self.subjects[subject_name].movement_sequence[trial_number]
        trial_name = movements[0].trial_name
        source, destination = self.trial_configuration.get_source_destination_pair_by_name(trial_name)
        return source, destination

    def _load_xy(self, subject_name: str, trial_number: int) -> \
            Optional[Tuple[np.ndarray, np.ndarray, list, np.ndarray, np.ndarray, list]]:
        """
        Load x, y coordinates for given subject and trial number.

        :param subject_name: The subject name.
        :param trial_number: The trial number for the subject.
        :return: A tuple containing the original x, y coordinates and the optimized x, y coordinates as numpy arrays.
        :rtype: np.ndarray, np.ndarray, np.ndarray, np.ndarray
        """
        subject = self.subjects[subject_name]
        self.current_subject = subject

        movements = subject.movement_sequence[trial_number]
        trial_name = movements[0].trial_name

        source, destination = self.trial_configuration.get_source_destination_pair_by_name(trial_name)
        _, shortcut = self.shortcut_map.get_shortest_path(source, destination)
        start, end = shortcut[0], shortcut[-1]

        continuous_movements = [{"pos": self.grid.get_map_pos(move.get_vector()), "data": move} for move in
                                movements]

        # convert to 0 indexed
        discrete_movements = [{"pos": tuple_sub_one(self.grid.get_block_pos(move.get_vector())), "data": move} for
                              move in
                              movements]

        # remove consecutive duplicates
        discrete_movements = [discrete_movements[i] for i in range(len(discrete_movements)) if
                              i == 0 or
                              (not np.array_equal(discrete_movements[i]["pos"],
                                                  discrete_movements[i - 1]["pos"])) and
                              not self.shortcut_map.check_coord_is_wall(discrete_movements[i]["pos"])]

        fraction_length = len(discrete_movements) // 3

        # check if start point is in first third of movements array
        discrete_coords = list(map(lambda m: m["pos"], discrete_movements))
        if start in discrete_coords[:fraction_length]:
            # remove all movements before start
            discrete_movements = discrete_movements[discrete_coords.index(start):]

        else:
            # add start to movements
            first_move_data = discrete_movements[0]["data"]
            discrete_movements = [{"pos": start, "data": first_move_data}] + discrete_movements

        # check offset is dot product not equal to 1,
        # then add paths from get_shortest_path_from_two_coords to interpolate
        i = 0

        while i < len(discrete_movements) - 1:

            offset = np.array(discrete_movements[i + 1]["pos"]) - np.array(discrete_movements[i]["pos"])
            if offset.dot(offset) > 1:
                shortcut = self.shortcut_map.get_shortest_path_from_two_coords(discrete_movements[i]["pos"],
                                                                               discrete_movements[i + 1]["pos"])

                last_move_data = discrete_movements[i]["data"]
                discrete_movements = discrete_movements[:i + 1] + list(
                    map(lambda p: {"pos": p, "data": last_move_data}, shortcut[1:-1])) + discrete_movements[i + 1:]
                i += len(shortcut) - 2
            else:
                i += 1

        # check if end point is in last third of movements array
        # if end in movements[-fraction_length:]:

        discrete_coords = list(map(lambda m: m["pos"], discrete_movements))
        if end in discrete_coords[-fraction_length:]:
            # remove all the elements after end
            discrete_movements = discrete_movements[:discrete_coords.index(end) + 1]
        else:
            # get the shortest path from last element to end
            shortcut = self.shortcut_map.get_shortest_path_from_two_coords(discrete_movements[-1]["pos"], end)
            if len(shortcut) - 2 < 3:
                last_move_data = discrete_movements[-1]["data"]
                discrete_movements = discrete_movements + list(
                    map(lambda p: {"pos": p, "data": last_move_data}, shortcut[1:]))

        discrete_movements = [{"pos": tuple_add(m["pos"], 0.5), "data": m["data"]} for m in discrete_movements]
        discrete_path = np.array(list(map(lambda m: m["pos"], discrete_movements)))

        x = (discrete_path[:, 0])
        y = (discrete_path[:, 1])

        previous_tuple = np.array([-1, -1])
        clean_continuous_movements = []
        for movement in continuous_movements:
            if not np.array_equal(movement["pos"], previous_tuple):
                clean_continuous_movements.append(movement)
                previous_tuple = movement["pos"]

        continuous_path = np.array(list(map(lambda m: m["pos"], clean_continuous_movements)))
        ox = (continuous_path[:, 0])
        oy = (continuous_path[:, 1])

        # add to cache
        self.cache[f"{subject_name},{trial_number}"] = ox, oy, clean_continuous_movements, x, y, discrete_movements

        return ox, oy, continuous_movements, x, y, discrete_movements

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

    def export_processed_data(self, subject_name, start=3, end=23, folder="processed_data"):
        """
        Export the processed data to a folder.
        :param subject_name: The name of the subject.
        :param start: The start index of the trial.
        :param end: The end index of the trial.
        :param folder: The folder to export to.
        :return: None
        """

        # create folder if not exist
        if not os.path.exists(folder):
            os.makedirs(folder)

        import csv
        header1 = ["SubjectName", "TrialNumber", "TrialName", "Time", "X", "Y", "Rotation"]
        header2 = ["SubjectName", "TrialNumber", "TrialName", "Time", "X", "Y", "Rotation"]
        # open or create file in the folder
        with open(folder + "/" + subject_name + ".csv", "w") as f:
            # create csv writer
            writer = csv.writer(f)
            # write header
            writer.writerow(header1)

            for i in range(start, end):
                _, _, continuous_movements, _, _, _ = self._get_cache(subject_name, i)
                # create a file in the folder
                for movement in continuous_movements:
                    writer.writerow([subject_name, movement["data"].trial_number, movement["data"].trial_name,
                                     movement["data"].trial_time, movement["data"].x, movement["data"].y,
                                     movement["data"].rotation])
                    pass

        with open(folder + "/" + subject_name + "_discrete.csv", "w") as f:
            # create csv writer
            writer = csv.writer(f)
            # write header
            writer.writerow(header2)

            for i in range(start, end):
                _, _, _, _, _, discrete_movements = self._get_cache(subject_name, i)
                # create a file in the folder
                for movement in discrete_movements:
                    x, y = movement["pos"]
                    writer.writerow([subject_name, movement["data"].trial_number, movement["data"].trial_name,
                                     movement["data"].trial_time, x, y,
                                     movement["data"].rotation])
                    pass

    def export_processed_data_for_these_subjects(self, subject_names, start=3, end=23, folder="processed_data"):
        for subject_name in subject_names:
            self.export_processed_data(subject_name, start, end, folder)

    def export_processed_data_for_all_subjects(self, start=3, end=23, folder="processed_data"):
        for subject_name in self.subjects.keys():
            self.export_processed_data(subject_name, start, end, folder)

    def _get_cache(self, subject_name: str, trial_number: int) -> Tuple[
        np.ndarray, np.ndarray, List[Dict[str, MovementData]], np.ndarray, np.ndarray, List[Dict[str, MovementData]]]:
        """
        Get the cached data for a given subject and trial number.
        :param subject_name: The name of the subject.
        :param trial_number: The trial number.
        :return: The cached data.
        :rtype: tuple
        """
        cache = self.cache.get(f"{subject_name},{trial_number}")
        if cache is None:
            return self._load_xy(subject_name, trial_number)
        return cache

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

    def calculate_efficiency_for_one_subject(self, subject_name: str, start: int, end: int, use_cache=True) \
            -> Dict[int, float]:
        """
        Calculate efficiency for one subject between the given trial range.

        :param subject_name: The name of the subject.
        :param start: The starting trial number (inclusive).
        :param end: The ending trial number (exclusive).
        :param use_cache: If True, use the cache to load data.
        :return: A dictionary with trial numbers as keys and efficiency values as values.
        :rtype: Dict[int, float]
        """
        efficiency_dict = {}
        for n in range(start, end):
            x = []
            if use_cache:
                _, _, _, x, _, _ = self._get_cache(subject_name, n)
            else:
                _, _, _, x, _, _ = self._load_xy(subject_name, n)
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

    def plot_for_these_subjects(self, subjects: List[str], start: int = 3, end: int = 23, save_only: bool = False,
                                use_cache=True):
        """
        Plot movement paths for a list of subjects between the given trial range.

        :param subjects: A list of subject names.
        :param start: The starting trial number (inclusive).
        :param end: The ending trial number (exclusive).
        :param save_only: If True, the plot will only be saved and not displayed.
        :param use_cache: If True, use the cache to load data.
        """
        for subject in subjects:
            for n in range(start, end):
                if use_cache:
                    ox, oy, _, x, y, _ = self._get_cache(subject, n)
                else:
                    ox, oy, _, x, y, _ = self._load_xy(subject, n)
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

    def calculate_frechet_for_one_subject(self, subject_name, start=3, end=23, use_cache=True):
        """
        Calculate Frechet distance for one subject between the given trial range.

        :param subject_name: The name of the subject.
        :param start: The starting trial number (inclusive).
        :param end: The ending trial number (exclusive).
        :param use_cache: If True, use the cache to load data.
        """

        distances = {}

        for n in range(start, end):
            if use_cache:
                _, _, _, x, y, _ = self._get_cache(subject_name, n)
            else:
                _, _, _, x, y, _ = self._load_xy(subject_name, n)

            # get name of the start and end points
            source, destination = self.get_source_destination(subject_name, n)

            # combine x and y into a single list of tuples
            # pos in p and q here are 0-indexed
            p = list(zip(x - 0.5, y - 0.5))
            _, q = self.learning_map.get_shortest_path(source, destination)
            _, r = self.shortcut_map.get_shortest_path(source, destination)

            # calculate Frechet distance
            import similaritymeasures
            distances[n] = {"learn": similaritymeasures.frechet_dist(p, q),
                            "shortcut": similaritymeasures.frechet_dist(p, r)}

        return distances
        pass

    def calculate_frechet_for_these_subjects(self, subjects: List[str], start=3, end=23, use_cache=True):
        """
        Calculate Frechet distance for a list of subjects between the given trial range.

        :param subjects: A list of subject names.
        :param start: The starting trial number (inclusive).
        :param end: The ending trial number (exclusive).
        :param use_cache: If True, use the cache to load data.
        """
        distances = {}
        for subject in subjects:
            distances[subject] = self.calculate_frechet_for_one_subject(subject, start, end, use_cache)
        return distances

    def calculate_frechet_for_all_subjects(self, start=3, end=23, excluding=None, use_cache=True):
        """
        Calculate Frechet distance for all subjects between the given trial range.

        :param start: The starting trial number (inclusive).
        :param end: The ending trial number (exclusive).
        :param excluding: A list of subject names to be excluded from the calculation.
        :param use_cache: If True, use the cache to load data.
        """
        if excluding is None:
            excluding = []

        subjects = [subject for subject in self.subjects if subject not in excluding]

        return self.calculate_frechet_for_these_subjects(subjects, start, end, use_cache)

    def export_distance_summary(self, subject_name, start=3, end=23, folder="distance", use_cache=True):
        """
        Export the distance summary for one subject between the given trial range.

        :param subject_name: The name of the subject.
        :param start: The starting trial number (inclusive).
        :param end: The ending trial number (exclusive).
        :param folder: The folder to save the summary file.
        :param use_cache: If True, use the cache to load data.
        """

        import os
        if not os.path.exists(folder):
            os.makedirs(folder)

        distances = self.calculate_frechet_for_one_subject(subject_name, start, end, use_cache)
        header = ["SubjectName", "TrialNumber", "FrechetLearn", "FrechetShortcut", "LearnDistance", "ShortcutDistance"]
        with open(f"{folder}/distance_summary_{subject_name}.csv", "w") as f:
            # csv writer
            import csv
            writer = csv.writer(f)
            writer.writerow(header)
            for n in range(start, end):
                source, destination = self.get_source_destination(subject_name, n)
                learn_distance = self.learning_map.get_shortest_distance(source, destination)
                shortcut_distance = self.shortcut_map.get_shortest_distance(source, destination)
                writer.writerow([subject_name, n, distances[n]["learn"], distances[n]["shortcut"], learn_distance,
                                 shortcut_distance])

    def export_distance_summary_for_these_subjects(self, subjects: List[str], start=3, end=23, folder="distance",
                                                   use_cache=True):
        """
        Export the distance summary for a list of subjects between the given trial range.

        :param subjects: A list of subject names.
        :param start: The starting trial number (inclusive).
        :param end: The ending trial number (exclusive).
        :param folder: The folder to save the summary file.
        :param use_cache: If True, use the cache to load data.
        """
        for subject in subjects:
            print(f"Exporting distance summary for {subject}...")
            self.export_distance_summary(subject, start, end, folder, use_cache)

    def export_distance_summary_for_all_subjects(self, start=3, end=23, folder="distance", excluding=None,
                                                 use_cache=True):
        """
        Export the distance summary for all subjects between the given trial range.

        :param start: The starting trial number (inclusive).
        :param end: The ending trial number (exclusive).
        :param folder: The folder to save the summary file.
        :param excluding: A list of subject names to be excluded from the calculation.
        :param use_cache: If True, use the cache to load data.
        """
        if excluding is None:
            excluding = []

        subjects = [subject for subject in self.subjects.keys() if subject not in excluding]

        self.export_distance_summary_for_these_subjects(subjects, start, end, folder, use_cache)
