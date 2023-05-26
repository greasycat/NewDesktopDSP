from analysis import Loader
from analysis import MovementAnalyzer, RotationAnalyzer
import pandas as pd
from analysis import ShortcutMap
from analysis.strategies import topological_strategy
import random


def analyze():
    pd.set_option('display.max_column', 500)

    loader = Loader(data_dir="data", extra_dir="extra", image_dir="images")
    loader.load(learning=True)

    shortcut_map = ShortcutMap("extra/new_walls.csv", "extra/objects.csv", "extra/new_shortcuts.csv")
    learning_map = ShortcutMap("extra/learning_walls.csv", "extra/objects.csv", "extra/learning_shortcuts.csv", learning=True)

    # print(learning_map.get_learning_path("Piano", "Stove"))

    # Currently the movement analyzer can only handle normal (not alternative) trials, will add that later this week
    movement_analyzer = MovementAnalyzer(loader, shortcut_map=shortcut_map, learning_map=learning_map)

    # rotation analyzer gives the absolute angular error for each trial (both normal and alternative)
    # rotation_analyzer = RotationAnalyzer(loader)

    # excluding = ["CY4GO", "PE12LE", "JU11SI", "LU24FR"]

    # random.seed(0)
    # sample = loader.sample_subject(10)
    # subject_names = [subject.name for subject in sample]

    # efficiencies = movement_analyzer.calculate_efficiency_for_all_subjects()
    # print(efficiencies)

    # failures = movement_analyzer.calculate_failure_for_all_subjects()
    # print(failures)

    # errors = rotation_analyzer.calculate_estimation_error_for_all_subjects()
    # print(errors)

    # movement_analyzer.plots_for_all_subjects()
    # movement_analyzer.export_processed_data_for_all_subjects()

    frechet = movement_analyzer.calculate_frechet_for_all_subjects(start=3, end=5)
    print(frechet)



    # movement_analyzer.export_processed_data_for_all_subjects()
    # movement_analyzer.plot_all_topological_paths()


if __name__ == "__main__":
    analyze()
