from __future__ import annotations
import csv
import heapq
import numpy as np

learning_order = [
    "Stove",
    "Piano",
    "Trash Can",
    "Bookshelf",
    "Wheelbarrow",
    "Harp",
    "Well",
    "Chair",
    "Mailbox",
    "Telescope",
    "Plant",
    "Picnic Table",
]


def load_objects(file_name):
    objects = {}
    with open(file_name, "r") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            obj = (row[0], int(row[1]), int(row[2]))
            objects[obj[0]] = (obj[1], obj[2])

    print("ShortcutMap: Loaded {} objects".format(len(objects)))
    return objects


def transform_walls(input_file, output_file):
    with open(input_file, "r") as input_csv:
        reader = csv.reader(input_csv)
        next(reader)  # Skip the header row

        with open(output_file, "w", newline="") as output_csv:
            writer = csv.writer(output_csv)
            writer.writerow(["X", "Y"])  # Write header row to output file

            for row in reader:
                x, y = int(row[0]), int(row[1])
                new_x, new_y = x - 1, y - 1
                writer.writerow([new_x, new_y])


def generate_connectivity_matrix(n):
    matrix = np.zeros((n * n, n * n))
    for i in range(n):
        for j in range(n):
            current_index = i * n + j
            if i > 0:  # connect to block above
                above_index = (i - 1) * n + j
                matrix[current_index][above_index] = 1
            if i < n - 1:  # connect to block below
                below_index = (i + 1) * n + j
                matrix[current_index][below_index] = 1
            if j > 0:  # connect to block to the left
                left_index = i * n + (j - 1)
                matrix[current_index][left_index] = 1
            if j < n - 1:  # connect to block to the right
                right_index = i * n + (j + 1)
                matrix[current_index][right_index] = 1
            # connect to current block
            matrix[current_index][current_index] = 1
    return matrix


class ShortcutMap:
    def __init__(self, wall_file, object_file,
                 shortcut_output_file="extra/shortcut_paths.csv",
                 learning_output_file="extra/learning_paths.csv",
                 reverse_learning_output_file="extra/reverse_learning_paths.csv",
                 learning=False):
        self.map_width = 11
        self.map_height = 11
        self.connectivity_matrix = generate_connectivity_matrix(11)
        self.load_walls(wall_file)
        self.objects = load_objects(object_file)
        self.shortest_paths = self.calculate_shortest_paths()
        self.learning_paths = None
        self.reverse_learning_paths = None
        if learning:
            self.learning_path = self.calculate_learning_path(learning_order)
            self.reverse_learning_paths = self.calculate_learning_path(learning_order[::-1])
            self.save_shortest_distances(learning_output_file, self.learning_path)
            self.save_shortest_distances(reverse_learning_output_file, self.reverse_learning_paths)
        else:
            self.save_shortest_distances(shortcut_output_file, self.shortest_paths)

    def calculate_learning_path(self, order):
        # get path for consecutive objects and handle the last object to first object case
        path = dict()
        for i in range(len(order)):
            path[order[i]] = dict()

        for i in range(len(order)):
            for j in range(len(order)):
                if i != j:
                    if i < j:
                        seq = order[i:j + 1]
                    else:
                        seq = order[i:] + order[:j + 1]

                    new_path = []
                    for k in range(len(seq) - 1):
                        # concatenate the paths
                        new_path += self.get_shortest_path(seq[k], seq[k + 1])[1]
                    # remove duplicates in new_path
                    new_path = list(dict.fromkeys(new_path))

                    path[order[i]][order[j]] = (len(new_path), new_path)
        return path

    def get_learning_path(self, start_object, end_object):
        return self.learning_path[start_object][end_object]

    def get_reverse_learning_path(self, start_object, end_object):
        return self.reverse_learning_paths[start_object][end_object]

    def get_shortest_path(self, start_object, end_object):
        try:
            return self.shortest_paths[start_object][end_object]
        except KeyError:
            pass
        try:
            distance, path = self.shortest_paths[end_object][start_object]
            return distance, path[::-1]
        except KeyError:
            return None

    def get_shortest_distance(self, start_object, end_object):
        return self.get_shortest_path(start_object, end_object)[0]

    def load_walls(self, file_name):
        with open(file_name, "r") as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            for row in reader:
                x, y = int(row[0]), int(row[1])
                current_index = self.coord_to_index((x, y))
                t = self.index_to_coord(current_index)
                # print("X: {}, Y: {}, Index: {}, t: {}".format(x, y, current_index, t))
                # set current block and neighboring blocks to 0
                self.connectivity_matrix[current_index][current_index] = 0
                if x > 0:  # set block left to 0
                    above_index = self.coord_to_index((x - 1, y))
                    self.connectivity_matrix[current_index][above_index] = 0
                    self.connectivity_matrix[above_index][current_index] = 0
                if x < self.map_width - 1:  # set block right to 0
                    below_index = self.coord_to_index((x + 1, y))
                    self.connectivity_matrix[current_index][below_index] = 0
                    self.connectivity_matrix[below_index][current_index] = 0
                if y > 0:  # set block to the left to 0
                    left_index = self.coord_to_index((x, y - 1))
                    self.connectivity_matrix[current_index][left_index] = 0
                    self.connectivity_matrix[left_index][current_index] = 0
                if y < self.map_height - 1:  # set block to the right to 0
                    right_index = self.coord_to_index((x, y + 1))
                    self.connectivity_matrix[current_index][right_index] = 0
                    self.connectivity_matrix[right_index][current_index] = 0

    def check_coord_is_wall(self, coord):
        index = self.coord_to_index(coord)
        return self.connectivity_matrix[index][index] == 0

    def coord_to_index(self, coord):
        x, y = coord
        return int(x + y * self.map_width)

    def index_to_coord(self, index):
        x = index % self.map_width
        y = (index - x) // self.map_width
        return x, y

    def dijkstra(self, start):
        n = len(self.connectivity_matrix)
        distances = [float('inf')] * n
        distances[start] = 0
        heap = [(0, start)]
        prev = [None] * n

        while heap:
            current_distance, current_node = heapq.heappop(heap)
            if current_distance > distances[current_node]:
                continue

            for neighbor in range(n):
                weight = self.connectivity_matrix[current_node][neighbor]
                if weight > 0:
                    distance = current_distance + weight
                    if distance < distances[neighbor]:
                        distances[neighbor] = distance
                        prev[neighbor] = current_node
                        heapq.heappush(heap, (distance, neighbor))

        return distances, prev

    def reconstruct_path(self, prev, end, coord=False):
        path = []
        current = end
        while current is not None:
            path.append(current)
            current = prev[current]
        path.reverse()
        if coord:
            return [self.index_to_coord(index) for index in path]
        return path

    def calculate_shortest_paths(self):
        shortest_paths = {}
        names = list(self.objects.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                start_name = names[i]
                end_name = names[j]
                start_obj = self.objects[start_name]
                end_obj = self.objects[end_name]
                start = (start_obj[0], start_obj[1])
                end = (end_obj[0], end_obj[1])
                # print("Calculating path from {} to {}".format(start, end))

                start_index = self.coord_to_index(start)
                end_index = self.coord_to_index(end)

                distances, prev = self.dijkstra(start_index)
                # add 1 to all distances to account for the fact that the distance is the number of blocks

                distances = [distance + 1 for distance in distances]

                distance = distances[end_index]
                path = self.reconstruct_path(prev, end_index, coord=True)

                nested_dict = {end_name: (distance, path)}
                if start_name in shortest_paths:
                    shortest_paths[start_name].update(nested_dict)
                else:
                    shortest_paths[start_name] = nested_dict

        return shortest_paths

    def get_all_objects_position(self):
        pos = []
        for obj in self.objects:
            pos.append(self.objects[obj])
        return pos

    def get_traversability_matrix(self):
        # create a grid map where all the walls are 0 and all the free space is 1
        traversability_matrix = np.ones((self.map_width, self.map_height))
        for x in range(self.map_width):
            for y in range(self.map_height):
                if self.check_coord_is_wall((x, y)):
                    traversability_matrix[x, y] = 0

        # add walls around the map
        # traversability_matrix = np.pad(traversability_matrix, pad_width=1, mode='constant', constant_values=0)

        return traversability_matrix

    def get_shortest_path_from_two_coords(self, coord1, coord2):
        index1 = self.coord_to_index(coord1)
        index2 = self.coord_to_index(coord2)
        distance, prev = self.dijkstra(index1)
        path = self.reconstruct_path(prev, index2, coord=True)
        return path

    def save_shortest_distances(self, file_name, paths):
        with open(file_name, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                ["Object1", "Object1_X", "Object1_Y", "Object2", "Object2_X", "Object2_Y", "Shortest_Distance", "Path"])

            for start_name in paths.keys():
                for end_name in paths[start_name].keys():
                    start_obj = self.objects[start_name]
                    end_obj = self.objects[end_name]
                    shortest_distance, path = paths[start_name][end_name]
                    path = "->".join([str(coord) for coord in path])
                    writer.writerow(
                        [start_name, start_obj[0], start_obj[1], end_name, end_obj[0], end_obj[1],
                         shortest_distance, path])


if __name__ == "__main__":
    print(generate_connectivity_matrix(3))
