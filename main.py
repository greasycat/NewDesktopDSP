from analysis import Loader
from analysis import MovementAnalyzer, RotationAnalyzer
import pandas as pd
from analysis import ShortcutMap
from analysis.constants import Constants
from analysis.strategies import topological_strategy, Strategy
import random


def analyze():
    pd.set_option('display.max_column', 500)

    constants_2 = Constants(alt=True)

    loader_2 = Loader(constants_2, data_dir="data_alt", extra_dir="extra", image_dir="images")
    loader_2.load(learning=False, force=True)

    # shortcut_map_1 = ShortcutMap("extra/new_walls_1.csv", "extra/objects_1.csv", "new_shortcuts_1.csv",)
    # learning_map_2 = ShortcutMap("extra/learning_walls_1.csv", "extra/objects_1.csv", "learning_shortcuts_1.csv",
    #                              "reverse_learning_path_1.csv",
    #                              learning=True)
    shortcut_map_2 = ShortcutMap(constants_2,
                                 wall_file="extra/shortcut_walls_2.csv",
                                 object_file="extra/objects_2.csv",
                                 shortcut_output_file="shortcut_path_2.csv", )
    learning_map_2 = ShortcutMap(constants_2,
                                 wall_file="extra/learning_walls_2.csv",
                                 object_file="extra/objects_2.csv",
                                 learning_output_file="learning_path_2.csv",
                                 reverse_learning_output_file="reverse_learning_path_2.csv",
                                 learning=True)

    strategy_2 = Strategy("extra/strategy_map_2.txt", "extra/strategy_landmarks_2.txt")

    # Currently the movement analyzer can only handle normal (not alternative) trials, will add that later this week
    movement_analyzer_2 = MovementAnalyzer(loader_2, shortcut_map=shortcut_map_2, learning_map=learning_map_2,
                                           strategy=strategy_2)

    # rotation analyzer gives the absolute angular error for each trial (both normal and alternative)
    # rotation_analyzer = RotationAnalyzer(loader)

    # excluding = ["CY4GO", "PE12LE", "JU11SI", "LU24FR"]

    # random.seed(0)
    # sample = loader.sample_subject(10)
    # subject_names = [subject.name for subject in sample]
    #
    # efficiencies = movement_analyzer_2.calculate_efficiency_for_all_subjects()
    # print(efficiencies)
    #
    # failures = movement_analyzer_2.calculate_failure_for_all_subjects()
    # print(failures)
    #
    # movement_analyzer_2.plots_for_all_subjects()
    # movement_analyzer_2.export_processed_data_for_all_subjects()

    # print(movement_analyzer_2.calculate_frechet_for_one_subject("A", start=3, end=23))

    frechet = movement_analyzer_2.calculate_frechet_for_all_subjects()
    movement_analyzer_2.export_distance_summary("distance_summary_2.csv")
    # print(frechet)


if __name__ == "__main__":
    analyze()
