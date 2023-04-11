from analysis import Loader
from analysis import MovementAnalyzer, RotationAnalyzer
import pandas as pd
from analysis import ShortcutMap
from analysis.shortcut_map import transform_walls
import random


def analyze():
    pd.set_option('display.max_column', 500)

    loader = Loader(data_dir="alldata", extra_dir="extra", image_dir="images")
    loader.load(learning=True)

    shortcut_map = ShortcutMap("extra/new_walls.csv", "extra/objects.csv", "extra/new_shortcuts.csv")

    # Currently the movement analyzer can only handle normal (not alternative) trials, will add that later this week
    movement_analyzer = MovementAnalyzer(loader, shortcut_map=shortcut_map)

    # rotation analyzer gives the absolute angular error for each trial (both normal and alternative)
    rotation_analyzer = RotationAnalyzer(loader)

    # excluding = ["CY4GO", "PE12LE", "JU11SI", "LU24FR"]

    random.seed(0)
    sample = loader.sample_subject(1)
    subject_names = [subject.name for subject in sample]

    efficiencies = movement_analyzer.calculate_efficiency_for_these_subjects(subject_names)
    print(efficiencies)

    failures = movement_analyzer.calculate_failure_for_these_subjects(subject_names)
    print(failures)

    errors = rotation_analyzer.calculate_estimation_error_for_one(subject_names[0])
    print(errors)

    # movement_analyzer.plot_for_these_subjects(subject_names, save_only=False)


if __name__ == "__main__":
    analyze()
