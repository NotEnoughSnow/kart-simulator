import ast
import csv


def readTrackFile(name):
    with open(name, "r") as f:
        reader = csv.reader(f, delimiter=",")
        shapes = list(reader)
        shapes_arr = [list(map(ast.literal_eval, shape)) for shape in shapes]

    return shapes_arr


def normalize_vec(vec, maximum=None, minimum=None):

    if maximum is None or minimum is None:
        assert maximum is None and minimum is None
        maximum = max(vec)
        minimum = min(vec)

    max_val = max(abs(maximum), abs(minimum))
    maximum = max_val
    minimum = -max_val

    return [2 * ((x - minimum) / (maximum - minimum)) - 1 for x in vec]
