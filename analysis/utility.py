from __future__ import annotations
# zero indexed tuple to one indexed tuple
def tuple_add_one(t):
    return tuple(map(lambda x: x + 1, t))


def tuple_sub_one(t):
    return tuple(map(lambda x: x - 1, t))


def tuple_add(t, value):
    return tuple(map(lambda x: x + value, t))


def list_add_one(li):
    return list(map(lambda y: tuple(map(lambda x: x + 1, y)), li))


def list_sub_one(li):
    return list(map(lambda y: tuple(map(lambda x: x - 1, y)), li))
