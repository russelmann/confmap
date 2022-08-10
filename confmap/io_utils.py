#  MIT License
#  Copyright (c) 2022. Ruslan Guseinov.

import time
from collections import deque
from typing import Tuple

import numpy as np
from numpy import typing as npt


class timer:
    name: str
    time_start: float

    def __init__(self, name: str = None):
        self.name = name

    def __enter__(self):
        self.time_start = time.time()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        message = f'[{self.name}] Elapsed: {time.time() - self.time_start:.6f} sec.'
        print(message)


def read_obj(file_path: str) -> Tuple[npt.NDArray, npt.NDArray]:
    """Read an OBJ file.

    :param file_path: File path.
    :return: NumPy array of vertex coordinates and NumPy array of vertex indices per face.
    """
    vertices, faces = deque(), deque()
    with open(file_path) as fin:
        for line in fin.readlines():
            if not line or not (words := line.split()):
                continue
            if words[0] == 'v':
                vertices.append(list(map(float, words[1:])))
            elif words[0] == 'f':
                faces.append(list(map(int, words[1:])))
    return np.array(vertices), np.array(faces) - 1


def write_obj(file_path: str,
              vertices: npt.ArrayLike,
              faces: npt.ArrayLike,
              uvs: npt.ArrayLike = None,
              uv_faces: npt.ArrayLike = None):
    """Write an OBJ file.

    :param file_path: File path.
    :param vertices: Array of vertices.
    :param faces: Array of faces.
    :param uvs: Array of UV coordinates.
    :param uv_faces: Array of UV faces.
    :return: None.
    """
    with open(file_path, 'wt') as fout:
        for vertex in vertices:
            fout.write(f'v {vertex[0]:.15} {vertex[1]:.15} {vertex[2]:.15}\n')
        if uvs is None:
            for face in faces:
                f0, f1, f2 = face + 1
                fout.write(f'f {f0} {f1} {f2}\n')
        else:
            for uv in uvs:
                fout.write(f'vt {uv[0]:.15} {uv[1]:.15}\n')
            for face, uv_face in zip(faces, uv_faces):
                f0, f1, f2 = face + 1
                g0, g1, g2 = uv_face + 1
                fout.write(f'f {f0}/{g0} {f1}/{g1} {f2}/{g2}\n')
