import numpy as np
import os
import pandas as pd
from scipy.spatial.transform import Rotation as R


def process_input(input_file):
    assert os.path.isfile(input_file)
    coords = pd.read_csv(input_file, sep="\s+", header=None, usecols=[1, 2, 3]).values
    assert coords.shape == (
        3,
        3,
    ), f"Not enough input {coords.shape}, check the input file."
    return coords


def rotation_matrix_from_vectors(vec1, vec2):
    """Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (
        vec2 / np.linalg.norm(vec2)
    ).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix


def get_distance(atom1, atom2):
    return np.sqrt(np.sum((atom1 - atom2) ** 2, axis=0))


def move_atom(data):
    connecting_vector = np.subtract(
        data[0], data[1]
    )  # vector connecting the center atom and the moving atom
    rot_matrix = rotation_matrix_from_vectors(
        connecting_vector, np.array([1, 0, 0])
    )  # Rotation between x axis and the vector connecting the atoms
    r = R.from_matrix(rot_matrix)  # Form the rotation matrix
    # Apply the rotation to all the vectors
    data[0] = r.apply(data[0])
    data[1] = r.apply(data[1])
    data[2] = r.apply(data[2])
    # find the distance between the center atom and the reference atom
    dist1 = get_distance(data[0], data[2])
    # find the distance between the center atom and the moving atom
    dist2 = get_distance(data[0], data[1])
    print(f"Distance between the center and reference atom: {dist1}")
    print(f"Distance between the center and moving atom: {dist2}")
    move_distance = dist2 - dist1
    print(f"Move distance: {move_distance}")
    # final_atom = r.apply(data[1]) - [move_distance, 0, 0]
    # apply inverse rotation to get the atom back in the previous coordinate system
    r2 = r.inv()
    data[0] = r2.apply(data[0])
    data[1] = r2.apply(data[1] + [move_distance, 0, 0])
    data[2] = r2.apply(data[2])
    dist3 = get_distance(data[0], data[1])
    print(f"Target distance: {dist1}; Final distance: {dist3}")
    print(f"New coordinates: {data}")


if __name__ == "__main__":
    input_file = "./input.csv"
    data = process_input(input_file)
    move_atom(data)