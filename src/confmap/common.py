#  MIT License
#  Copyright (c) 2022. Ruslan Guseinov.

from itertools import tee
from typing import Iterable

from numpy import typing as npt

IntArray = npt.NDArray[int]
FloatArray = npt.NDArray[float]


def pairwise(iterable: Iterable):
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)
