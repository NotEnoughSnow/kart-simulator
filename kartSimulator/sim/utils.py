import ast
import csv
import os

def get_next_run_directory(base_dir, experiment_type):
    """
    Returns the next run directory path for the given experiment type
    by incrementing the run number.
    """
    run_number = 1
    while True:
        run_path = os.path.join(base_dir, experiment_type, f'ver_{run_number}')
        if not os.path.exists(run_path):
            os.makedirs(run_path)
            return run_path, run_number
        run_number += 1

def get_next_run_directory_mod(base_dir, experiment_type):
    """
    Returns the next run directory path for the given experiment type
    by incrementing the run number.
    """
    run_number = 1
    while True:
        run_path = os.path.join(base_dir, f"{experiment_type}-{run_number}")
        if not os.path.exists(run_path):
            os.makedirs(run_path)
            return run_path, run_number
        run_number += 1


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
