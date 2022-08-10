from tests import TEST_DATA_FOLDER
from confmap.mesh_utils import compute_mesh_edges, face_face_adjacency
import numpy as np
import os.path
from confmap.io_utils import read_obj


def test_boundary_loops():
    file_path = os.path.join(TEST_DATA_FOLDER, 'ico.obj')
    vertices, faces = read_obj(file_path)
    _, _, boundary_loops = compute_mesh_edges(faces)
    assert len(boundary_loops) == 2
    boundary_loops = [np.roll(loop, shift=-np.argmin(loop)) for loop in boundary_loops]
    loop_a, loop_b = boundary_loops
    if len(loop_b) < len(loop_a):
        loop_a, loop_b = loop_b, loop_a
    assert (loop_a == [9, 35, 29]).all()
    assert (loop_b == [0, 11, 2, 24, 23, 26, 17, 16]).all()


def test_face_face_adjacency():
    file_path = os.path.join(TEST_DATA_FOLDER, 'ico.obj')
    _, faces = read_obj(file_path)
    ff, ffi = face_face_adjacency(faces)
    assert ((ff == -1) | (ff[ff, ffi] == np.arange(len(ff))[:, np.newaxis])).all()
