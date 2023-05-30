from __future__ import annotations
from pathlib import Path

from analysis.constants import Constants
from analysis.subject import Subject
from analysis.movement_data import MovementData
from analysis.rotation_data import RotationData
from analysis.trial_configuration import TrialConfiguration
import csv

from typing import List, Dict, Iterator, Union, Tuple

MOVEMENT_FILE = 'movement.csv'
ROTATION_FILE = 'rotation.csv'
TIMEOUT_FILE = 'timeout.txt'
META_FILE = 'meta.txt'


class CorruptedDataError(Exception):
    pass


class InsufficientDataError(Exception):
    pass


class Loader:
    def __init__(self, constants: Constants, data_dir: str = 'data', extra_dir: str = "extra",
                 image_dir: str = "images", ):
        """
        Loader class for loading subjects, movements, and rotations data.

        :param data_dir: Directory containing the data files.
        :param extra_dir: Directory containing the extra files like walls, shortcuts, and trial configuration.
        :param image_dir: Directory containing the images.
        """
        self.root_dir = Path(data_dir)
        self.extra_dir = Path(extra_dir)
        self.image_dir = Path(image_dir)
        self.subjects: Dict[str, Subject] = {}
        self.shortcuts = None
        self.trial_configuration = None
        self.image_maze1 = None
        self.constants = constants

    def get_subjects(self, subjects: List[str]) -> List[Subject]:
        """
        Get a list of Subject objects based on the provided subject names.

        :param subjects: A list of subject names.
        :return: A list of Subject objects.
        """
        return [self.subjects[subject] for subject in subjects]

    def load(self, force: bool = False, learning: bool = False):
        """
        Load subjects, movements, and rotations data.

        :param force: Whether to raise an exception if there are insufficient data files.
        :param learning: Whether to include learning trials.
        :param image_file: Name of the image file containing the maze.
        """

        if not self.root_dir:
            return

        # Load Trial Configuration
        self.trial_configuration = TrialConfiguration(
            yield_csv_todict(self.extra_dir.joinpath(self.constants.trial_file)))

        self.image_maze1 = self.image_dir.joinpath(self.constants.maze_image)

        # Get participants dirs
        for participant_dir in self.root_dir.iterdir():
            file_paths = {}

            # Skip if it is a file
            if participant_dir.is_file():
                continue

            for sub_file in participant_dir.iterdir():
                if sub_file.is_dir():
                    continue
                file_paths[sub_file.name] = sub_file

            if len(file_paths.items()) != 4 and not force:
                raise InsufficientDataError("Corrupted file")

            subject = Subject(name=participant_dir.name)
            try:
                subject.meta = load_meta(file_paths[META_FILE])
            except KeyError:
                if not force:
                    raise CorruptedDataError(
                        "Make sure you have the correct filename for the meta file, otherwise exclude the folder" + str(
                            participant_dir))

                print("Cannot load meta file, skipping")
            try:
                subject.rotation_sequence = load_rotation(file_paths[ROTATION_FILE])
            except KeyError:
                if not force:
                    raise CorruptedDataError(
                        "Make sure you have the correct filename for the rotation file, otherwise exclude the folder" + str(
                            participant_dir))
                else:
                    print("Cannot load rotation file, skipping")

            try:
                subject.movement_sequence = load_movement(file_paths[MOVEMENT_FILE], learning=learning)
            except KeyError:
                if not force:
                    raise CorruptedDataError(
                        "Make sure you have the correct filename for the movement file, otherwise exclude the folder" + str(
                            participant_dir))

                print("Cannot load movement file, skipping")

            try:
                subject.timeout_trials = load_timeout(file_paths[TIMEOUT_FILE])
            except KeyError:
                if not force:
                    raise CorruptedDataError(
                        "Make sure you have the correct filename for the timeout file, otherwise exclude the folder" + str(
                            participant_dir))

                print("Cannot load timeout file, skipping")

            self.subjects[participant_dir.name] = subject

    def sample_subject(self, n: int = 5) -> List[Subject]:
        """
        Randomly sample n subjects.

        :param n: The number of subjects to sample.
        :return: A list of sampled Subject objects.
        """
        import random
        keys = random.sample(list(self.subjects.keys()), n)
        return [self.subjects[key] for key in keys]


def load_rotation(path):
    rotation_trials = {}
    with open(path, newline='', encoding="utf-8-sig") as csvfile:
        dict_reader = csv.DictReader(csvfile, delimiter=',', skipinitialspace=True)
        for row in dict_reader:
            rotation_trials[int(row["Trial"])] = RotationData.from_dict(row)
    return rotation_trials


def load_csv_tolist(path):
    """:return list(list(str))"""
    result = []
    with open(path, newline='', encoding="utf-8-sig") as csvfile:
        reader = csv.reader(csvfile, delimiter=',', skipinitialspace=True)
        next(reader)
        for row in reader:
            result.append(row)
    return result


def yield_csv_todict(path: Path):
    with open(path, newline='', encoding="utf-8-sig") as csvfile:
        dict_reader = csv.DictReader(csvfile, delimiter=",", skipinitialspace=True)
        for row in dict_reader:
            yield row


def load_movement(path, learning=False):
    """:return dict(int:list)"""
    movement_trials = {}
    with path.open("r") as file:
        last_trial_number = 0
        skip_header_line = False
        for line in file:
            maker_pos = line.find("@")
            if skip_header_line:
                skip_header_line = False
                continue
            if line.find("@") != -1:
                last_trial_number = int(line[maker_pos + 1:])
                movement_trials[last_trial_number] = []
                skip_header_line = True
            else:
                movement_trials[last_trial_number].append(MovementData.from_str(line))
    if not learning and 99 in movement_trials:
        movement_trials.pop(99)
    return movement_trials


def load_timeout(path):
    timeout_trials = []
    with open(path, newline='', encoding="utf-8-sig") as csvfile:
        # try int parse the first line, if it fails, it is a header, otherwise, it is not a header
        try:
            int(csvfile.readline())
            csvfile.seek(0)
        except ValueError:
            pass
        reader = csv.reader(csvfile, delimiter=',', skipinitialspace=True)
        for row in reader:
            timeout_trials.append(int(row[0]))

    return timeout_trials


def load_meta(path):
    """:return dict(str:str)"""
    meta_dict = {}
    with path.open("r") as file:
        for line in file:
            pair = [element.strip() for element in line.split(":")]
            if len(pair) != 2:
                continue
            meta_dict[pair[0]] = pair[1]
    return meta_dict
