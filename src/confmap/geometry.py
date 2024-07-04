#  MIT License
#  Copyright (c) 2022. Ruslan Guseinov.

from abc import ABC, abstractmethod

import numpy as np

from confmap.common import FloatArray
from confmap.comp_utils import compute_exterior_angles_2d


class ParametricCurve(ABC):
    """Arc-length parameterized curve."""

    @abstractmethod
    def discretize(self, s: FloatArray, normalized: float = True) -> FloatArray:
        """Discretize curve by given length offsets from the curve's origin.

        :param s: Offsets.
        :param normalized: Using normalized curve length (to be equal to one).
        :return: Discretization points.
        """
        pass

    def compute_exterior_angles(self, s: FloatArray, normalized: float = True) -> FloatArray:
        """Compute exterior angles when curve is discretize by given length offsets from the curve's origin.

        :param s: Offsets.
        :param normalized: Using normalized curve length (to be equal to one).
        :return: Exterior angles.
        """
        return compute_exterior_angles_2d(self.discretize(s, normalized))


class ParametricCircle(ParametricCurve):

    def __init__(self, radius: float = 1.):
        self._radius = radius

    def discretize(self, s: FloatArray, normalized: float = True) -> FloatArray:
        if normalized:
            s = 2. * np.pi * s
        points = np.empty((len(s), 2))
        points[:, 0], points[:, 1] = np.cos(s), np.sin(s)
        points *= self._radius
        return points
