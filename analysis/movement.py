import numpy as np
from analysis.grid import Grid
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches


# H11 W11
# TopLeft 0.5, 225
# TopRight 227 225
# BottomLeft 0.5 -1
# BottomRight 227 -1
# for key, value in subjects.movement_sequence.items():

class Shortcuts:
    def __init__(self, dict_generator):
        self.shortcuts = {}
        for row in dict_generator:
            source = row["Source"]
            destination = row["Destination"]
            distance = float(row["Distance"])
            inner_dict = {destination: distance}
            if source in self.shortcuts.keys():
                self.shortcuts[source].update(inner_dict)
            else:
                self.shortcuts[source] = inner_dict

    def get_shortcut(self, a, b):
        try:
            return self.shortcuts[a][b]
        except KeyError:
            pass
        try:
            return self.shortcuts[b][a]
        except KeyError:
            return None


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
    def __init__(self, loader, origin=np.array([3, 0]), map_actual_size=np.array([225, 225]),
                 grid_size=np.array([11, 11])):
        self.grid = Grid(origin=origin, map_actual_size=map_actual_size, grid_size=grid_size)
        self.walls = loader.walls
        self.shortcut = loader.shortcuts
        self.trial_configuration = loader.trial_configuration
        self.current_subject = None
        self.bg_1 = loader.image_maze1
        self.subjects = loader.subjects
        pass

    def load_xy(self, subject, trial_number):
        """:return np.ndarray,np.ndarray"""
        try:
            self.current_subject = subject
            path = []
            last_moved_block = np.zeros(2)
            for move in subject.movement_sequence[trial_number]:
                current_pos = self.grid.get_block_pos(move.get_vector())
                if np.array_equal(last_moved_block, current_pos):
                    continue
                # Check offset
                offset = current_pos - last_moved_block
                if not np.array_equal(offset, current_pos) and offset.dot(offset) != 1:
                    correction_offset_1 = offset.copy()
                    correction_offset_2 = offset.copy()
                    correction_offset_1[0] = 0
                    correction_offset_2[1] = 0
                    correction_point_1 = correction_offset_1 + last_moved_block
                    correction_point_2 = correction_offset_2 + last_moved_block
                    if tuple(map(int, correction_point_1.tolist())) not in self.walls:
                        path.append(correction_point_1)
                    elif tuple(map(int, correction_point_2.tolist())) not in self.walls:
                        path.append(correction_point_2)
                    else:
                        pass

                path.append(current_pos)
                last_moved_block = current_pos

            arr = np.array(path)
            x = (arr[:, 0]) - 0.5
            y = (arr[:, 1]) - 0.5
            return x, y
        except:
           raise KeyError("Makesure you have the uncorrupted file, otherwise exclude the folder "+ str(subject.name))

    def draw(self, n, x, y, bg_file=""):

        if bg_file == "":
            bg_file = self.bg_1

        fig, ax = plt.subplots()
        ax.step(x, y,color='red', alpha=0.2)

        plt.autoscale(False)
        # print(bg_file)
        #bg = mpimg.imread("/Users/carolhe/Documents/Research/Fifth_Year_Projects/NavyGrant/NewDesktopDSP/images/maze1.png")
        bg = mpimg.imread(bg_file)

        plt.imshow(bg, extent=[0, self.grid.grid_size[0], 0, self.grid.grid_size[1]])

        if self.shortcut:
            pass

        trial_name = self.current_subject.movement_sequence[n][0].trial_name
        source, destination = self.trial_configuration.get_source_destination_pair_by_name(trial_name)
        shortest = self.shortcut.get_shortcut(source, destination)
        estimated_distance = len(x)-1
        efficiency = estimated_distance / shortest 
        if efficiency < 1:
            efficiency = 1
        plt.title(
            f"Trial {n-2}@{trial_name}\n "
            f"From {source} to {destination}\n "
            f"Distance: {estimated_distance} "
            f"Shortest: {shortest} "
            f"Efficiency: {efficiency:.2f}")
        plt.xticks(np.arange(0, self.grid.grid_size[0] + 1, step=1))
        plt.yticks(np.arange(0, self.grid.grid_size[1] + 1, step=1))
        plt.grid()
        fig.set_size_inches(7, 7)
        # plt.savefig('path_plot/'+subject+'_'+trial_name+'_Trial'+str(n-2)+'.png',dpi=80)
        plt.show()

    def calculate_efficiency(self, n, x, y):
        trial_name = self.current_subject.movement_sequence[n][0].trial_name
        if n in self.current_subject.timeout_trials:
            # print(trial_name)
            return 2.54
        else:
            source, destination = self.trial_configuration.get_source_destination_pair_by_name(trial_name)
            shortest = self.shortcut.get_shortcut(source, destination)
            estimated_distance = len(x)-1
            efficiency =  estimated_distance / shortest
            if efficiency < 1:
                efficiency = 1
            return efficiency

    def calculate_efficiency_for_one(self, subject, start, end):
        efficiency_dict = {}
        for n in range(start, end):
            x, y = self.load_xy(self.subjects[subject], n)
            # self.current_subject = self.subjects[subject]
            # print(self.current_subject.movement_sequence[n][0].trial_name)
            efficiency_dict[n] = self.calculate_efficiency(n, x, y)
        return efficiency_dict

    def calculate_efficiency_for_all(self, start=3, end=23, excluding=None):
        if excluding is None:
            excluding = []

        efficiency_dict = {}
        for subject in self.subjects:
            if subject in excluding:
                continue
            efficiency_dict[subject] = self.calculate_efficiency_for_one(subject, start, end)
        return efficiency_dict
        
    def calculate_failure_for_all(self, excluding=None):
        if excluding is None:
            excluding = []

        failure_dict = {}
        for subject in self.subjects:
            if subject in excluding:
                continue
            self.current_subject = self.subjects[subject]
            # print(self.current_subject.timeout_trials)
            failure_dict[subject] = len(self.current_subject.timeout_trials)
        return failure_dict
        
    def save_plots_for_all(self, start=3, end=23, excluding=None):
        if excluding is None:
            excluding = []
            
        for subject in self.subjects:
            if subject in excluding:
                continue
            for n in range(start, end):
                x, y = self.load_xy(self.subjects[subject], n)
                self.draw(subject = subject, n=n, x=x, y=y)
        pass
