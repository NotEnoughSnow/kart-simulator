import ast
import csv


def readTrackFile(name):
    with open(name, "r") as f:
        reader = csv.reader(f, delimiter=",")
        shapes = list(reader)
        shapes_arr = [list(map(ast.literal_eval, shape)) for shape in shapes]

    return shapes_arr