import os

import numpy as np

from confmap.io_utils import read_obj, write_obj
from tests import TEST_DATA_FOLDER


def test_read_obj():
    file_path = os.path.join(TEST_DATA_FOLDER, 'ico.obj')
    vertices, faces = read_obj(file_path)
    assert len(vertices) == 39
    assert len(faces) == 67


def test_write_obj():
    file_path = os.path.join(TEST_DATA_FOLDER, 'tmp_test.obj')
    vertices, faces = np.identity(3), np.array([[0, 1, 2]])
    write_obj(file_path, vertices, faces)
    os.remove(file_path)
