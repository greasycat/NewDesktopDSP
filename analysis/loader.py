from pathlib import Path
from analysis.subject import Subject
from analysis.movement import MovementData, Shortcuts
from analysis.rotation import RotationData
from analysis.trial_configuration import TrialConfiguration
import csv

MOVEMENT_FILE = 'movement.csv'
ROTATION_FILE = 'rotation.csv'
TIMEOUT_FILE = 'timeout.txt'
META_FILE = 'meta.txt'


class CorruptedDataError(Exception):
    pass


class InsufficientDataError(Exception):
    pass


class Loader:
    def __init__(self, data_dir='data', extra_dir="extra", image_dir="images"):
        self.root_dir = Path(data_dir)
        self.extra_dir = Path(extra_dir)
        self.image_dir = Path(image_dir)
        self.subjects = {}
        self.walls = []
        self.shortcuts = None
        self.trial_configuration = None
        self.image_maze1 = None

    def get_subjects(self, subjects):
        return [self.subjects[subject] for subject in subjects]

    def load(self, force=False, learning=False):
        if not self.root_dir:
            return

        # Load Walls
        for pair in load_csv_tolist(self.extra_dir.joinpath("walls.csv")):
            if len(pair) < 2:
                continue
            self.walls.append(tuple(map(int, pair)))

        # Load Shortcuts
        self.shortcuts = Shortcuts(yield_csv_todict(self.extra_dir.joinpath("shortcuts_1.csv")))

        # Load Trial Configuration
        self.trial_configuration = TrialConfiguration(yield_csv_todict(self.extra_dir.joinpath("trial_1.csv")))

        self.image_maze1 = self.image_dir.joinpath("maze1.png")

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

            if len(file_paths.items()) != 4 and force:
                raise InsufficientDataError("Corrupted file")

            subject = Subject(name=participant_dir.name)
            try:
                subject.meta = load_meta(file_paths[META_FILE])
                subject.rotation_sequence = load_rotation(file_paths[ROTATION_FILE])
                subject.movement_sequence = load_movement(file_paths[MOVEMENT_FILE], learning=learning)
                # TODO: add timeout here
                subject.timeout_trials = load_timeout(file_paths[TIMEOUT_FILE])
            except KeyError as e:
                # print(e)
                if not force:
                    raise CorruptedDataError("Makesure you have all the filenames correct, otherwise exclude the folder"+ str(participant_dir))

            self.subjects[participant_dir.name] = subject


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
        dict_reader = csv.DictReader(csvfile, delimiter=',', skipinitialspace=True)
        for row in dict_reader:
            timeout_trials.append(int(row["Trial"]))
    # print(timeout_trials)
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
