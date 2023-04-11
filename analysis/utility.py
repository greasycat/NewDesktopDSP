# zero indexed tuple to one indexed tuple
def tuple_add_one(t):
    return tuple(map(lambda x: x + 1, t))


def tuple_sub_one(t):
    return tuple(map(lambda x: x - 1, t))


def list_add_one(l):
    return list(map(lambda y: tuple(map(lambda x: x + 1, y)), l))


def list_sub_one(l):
    return list(map(lambda y: tuple(map(lambda x: x - 1, y)), l))
