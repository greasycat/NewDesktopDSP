# from analysis import ShortcutMap, Loader
#
# LEARN_ROUTE_OBJECTS = ["Trash Can",
#                        "Bookshelf",
#                        "Wheelbarrow",
#                        "Harp",
#                        "Well",
#                        "Chair",
#                        "Mailbox",
#                        "Telescope",
#                        "Plant",
#                        "Picnic Table",
#                        "Stove",
#                        "Piano",
#                        "Trash Can"]
#
# from frechetdist import frdist
#
#
# class Frechet:
#     def __init__(self, loader: Loader, shortcut_map: ShortcutMap):
#         self.loader = loader
#         self.shortcut_map = shortcut_map
#         self.learn_routes = self.get_learn_route_from_shortcut()
#
#     def get_learn_route_from_shortcut(self):
#         learn_route = []
#         for i in range(len(LEARN_ROUTE_OBJECTS)-1):
#             _, path = self.shortcut_map.get_shortest_path(LEARN_ROUTE_OBJECTS[i], LEARN_ROUTE_OBJECTS[(i + 1)])
#             learn_route.extend(path[:-1])
#         return learn_route
#
#     def frechet_distance(self, path1, path2):
#         frdist(path1, path2)
#         pass

