learning_order_1 = [
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

learning_order_2 = [
    "TV",
    "Desk",
    "Water cooler",
    "Streetlamp",
    "Stepladder",
    "Fridge",
    "Bicycle",
    "Statue",
    "Couch",
    "Phonebooth",
    "Swing",
    "Clock",
]
class Constants:
    def __init__(self, alt=False):
        self.trial_file = None
        self.maze_image = None
        self.origin = None
        self.strategy_map = None
        self.strategy_landmarks = None
        self.learning_order = None
        if alt:
            self.trial_file = "trial_2.csv"
            self.maze_image = "maze_2.png"
            self.origin = [0.5, -1.1]
            self.strategy_landmarks = "extra/strategy_landmarks_2.txt"
            self.strategy_map = "extra/strategy_map_2.txt"
            self.learning_order = learning_order_2
        else:
            self.trial_file = "trial_1.csv"
            self.maze_image = "maze_1.png"
            self.origin = [3, 0]
            self.strategy_landmarks = "extra/strategy_landmarks_1.txt"
            self.strategy_map = "extra/strategy_map_1.txt"
            self.learning_order = learning_order_1
