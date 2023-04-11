from typing import Dict, Any

import numpy as np
from analysis.grid import Grid
from analysis.shortcut_map import ShortcutMap
from analysis.utility import tuple_add_one, tuple_sub_one, list_sub_one, list_add_one
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from typing import List, Dict, Any, Optional, Generator
import os


# H11 W11
# TopLeft 0.5, 225
# TopRight 227 225
# BottomLeft 0.5 -1
# BottomRight 227 -1
# for key, value in subjects.movement_sequence.items():

# class Shortcuts:
#     def __init__(self, dict_generator):
#         self.shortcuts = {}
#         for row in dict_generator:
#             source = row["Source"]
#             destination = row["Destination"]
#             distance = float(row["Distance"])
#             inner_dict = {destination: distance}
#             if source in self.shortcuts.keys():
#                 self.shortcuts[source].update(inner_dict)
#             else:
#                 self.shortcuts[source] = inner_dict
#
#     def get_shortcut(self, a, b):
#         try:
#             return self.shortcuts[a][b]
#         except KeyError:
#             pass
#         try:
#             return self.shortcuts[b][a]
#         except KeyError:
#             return None


class MovementData:
    def __init__(self, trial_name, trial_number, trial_time, x, y, rotation):
        self.y = y
        self.x = x
        self.trial_time = trial_time
        self.trial_name = trial_name
        self.trial_number = trial_number
        self.rotation = rotation

    def get_vector(self):
        """:return np.ndarray"""
        return np.array([float(self.x), float(self.y)])

    @staticmethod
    def from_str(s=""):
        """:return MovementData"""
        elements = [element.strip() for element in s.split(",")]
        if len(elements) < 6:
            return None
        return MovementData(elements[0], elements[1], elements[2], elements[3], elements[4], elements[5])


class MovementAnalyzer:
    # def __init__(self, loader, origin=np.array([0.6, -1.1]), map_actual_size=np.array([226, 226]),
    #              grid_size=np.array([11, 11])):
    def __init__(self, loader, shortcut_map: ShortcutMap, origin=np.array([3, 0]), map_actual_size=np.array([225, 225]),
                 grid_size=np.array([11, 11])):
        self.grid = Grid(origin=origin, map_actual_size=map_actual_size, grid_size=grid_size)
        self.walls = loader.walls
        self.shortcut_map = shortcut_map
        self.trial_configuration = loader.trial_configuration
        self.current_subject = None
        self.bg_1 = loader.image_maze1
        self.subjects = loader.subjects
        pass

    def _load_xy(self, subject, trial_number):
        """:return np.ndarray,np.ndarray"""
        try:
            self.current_subject = subject

            movements = subject.movement_sequence[trial_number]
            trial_name = movements[0].trial_name

            source, destination = self.trial_configuration.get_source_destination_pair_by_name(trial_name)
            _, shortcut = self.shortcut_map.get_shortest_path(source, destination)
            start, end = shortcut[0], shortcut[-1]
            print(source, destination)
            print(start, end)
            print(shortcut)

            original_movement = [self.grid.get_map_pos(move.get_vector()) for move in movements]

            # convert to 0 indexed
            movements = list_sub_one([self.grid.get_block_pos(move.get_vector()) for move in movements])

            movements = [start] + movements


            # remove consecutive duplicates
            movements = [movements[i] for i in range(len(movements)) if
                         i == 0 or
                         (not np.array_equal(movements[i], movements[i - 1])) and
                         not self.shortcut_map.check_coord_is_wall(movements[i])]

            print(movements)

            # check offset is dot product not equal to 1,
            # then add paths from get_shortest_path_from_two_coords to interpolate
            i = 0
            is_end_in_movements = False
            where_is_the_end = 0
            while i < len(movements) - 1:

                # mark if the movement already contains an end
                if np.array_equal(movements[i + 1], end):
                    is_end_in_movements = True
                    where_is_the_end = i + 1

                offset = np.array(movements[i + 1]) - np.array(movements[i])
                if offset.dot(offset) > 1:
                    shortcut = self.shortcut_map.get_shortest_path_from_two_coords(movements[i], movements[i + 1])
                    movements = movements[:i + 1] + shortcut[1:-1] + movements[i + 1:]
                    i += len(shortcut) - 2
                else:
                    i += 1

            # remove all the elements after end
            if is_end_in_movements:
                movements = movements[:where_is_the_end + 1]
            else:
                # get the shortest path from last element to end
                shortcut = self.shortcut_map.get_shortest_path_from_two_coords(movements[-1], end)
                if len(shortcut) - 2 < 3:
                    movements = movements + shortcut[1:]

            path = list_add_one(movements)

            # add start and end to movement
            # for move in movements
            #     current_pos = self.grid.get_block_pos(move.get_vector())
            #     print(current_pos)
            #     if np.array_equal(last_moved_block, current_pos):
            #         continue
            #     # Check offset
            #     offset = current_pos - last_moved_block
            #     if not np.array_equal(offset, current_pos) and offset.dot(offset) != 1:

            # correction_offset_1 = offset.copy()
            # correction_offset_2 = offset.copy()
            # correction_offset_1[0] = 0
            # correction_offset_2[1] = 0
            # correction_point_1 = correction_offset_1 + last_moved_block
            # correction_point_2 = correction_offset_2 + last_moved_block
            # if tuple(map(int, correction_point_1.tolist())) not in self.walls:
            #     path.append(correction_point_1)
            # elif tuple(map(int, correction_point_2.tolist())) not in self.walls:
            #     path.append(correction_point_2)
            # else:
            #     pass

            # path.append(current_pos)
            # last_moved_block = current_pos

            arr = np.array(path)
            x = (arr[:, 0]) - 0.5
            y = (arr[:, 1]) - 0.5

            original_movement = np.array(original_movement)
            original_movement = original_movement[np.r_[True, np.any(np.diff(original_movement, axis=0), axis=1)]]

            return original_movement[:, 0], original_movement[:, 1], x, y
        except IndexError:
            raise Exception("Makesure you have the uncorrupted file, otherwise exclude the folder " + str(subject.name))
        except KeyError:
            raise Exception("Makesure you have the uncorrupted file, otherwise exclude the folder " + str(subject.name))

    def _draw(self, subject: str, n: int, ox, oy, x: List[float], y: List[float], bg_file: str = "",
              save_only: bool = False) -> None:
        """
        Draws a path plot for a given subject, trial, and path.

        Args:
            subject (str): The name of the subject.
            n (int): The index of the trial to draw.
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

    def _calculate_efficiency(self, n, x, y):
        trial_name = self.current_subject.movement_sequence[n][0].trial_name
        if n in self.current_subject.timeout_trials:
            return 2.54
        else:
            source, destination = self.trial_configuration.get_source_destination_pair_by_name(trial_name)
            shortest = self.shortcut_map.get_shortest_path(source, destination)[0]
            estimated_distance = len(x) - 1
            efficiency = estimated_distance / shortest
            if efficiency < 1:
                efficiency = 1
            return efficiency

    def calculate_efficiency_for_one_subject(self, subject, start, end):
        efficiency_dict = {}
        for n in range(start, end):
            _, _, x, y = self._load_xy(self.subjects[subject], n)
            efficiency_dict[n] = self._calculate_efficiency(n, x, y)
        return efficiency_dict

    def calculate_efficiency_for_these_subjects(self, subjects, start, end):
        efficiency_dict = {}
        for subject in subjects:
            efficiency_dict[subject] = self.calculate_efficiency_for_one_subject(subject, start, end)
        return efficiency_dict

    def calculate_failure_for_these_subjects(self, subjects):
        failure_dict = {}
        for subject in subjects:
            failure_dict[subject] = len(self.subjects[subject].timeout_trials)
        return failure_dict

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

    def calculate_failure_for_all_subjects(self, excluding=None):
        if excluding is None:
            excluding = []

        subjects = [subject for subject in self.subjects if subject not in excluding]
        return self.calculate_failure_for_these_subjects(subjects)

    def plot_for_these_subjects(self, subjects, start=3, end=23, save_only=False):
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
