#  MIT License
#  Copyright (c) 2022. Ruslan Guseinov.

from typing import Tuple, Union
from warnings import warn

import numpy as np
from numpy import typing as npt

from confmap.common import FloatArray


def quantize_value(value: npt.ArrayLike, quant: float) -> npt.ArrayLike:
    value = np.asarray(value)
    result = (value + quant / 2.) // quant * quant
    if result.ndim:
        result[result == 0] = quant
        result[np.isclose(result, 2. * np.pi)] = 2. * np.pi - quant
        return result
    else:
        if result == 0:
            return quant
        elif np.isclose(result, 2. * np.pi):
            return 2. * np.pi - quant
        return float(result)


_clausen1 = np.array([5.590566394715132269e-2, 0., 1.7630887438981157e-4, 0., 1.26627414611565e-6, 0.,
                      1.171718181344e-8, 0., 1.2300641288e-10, 0., 1.39527290e-12, 0., 1.669078e-14, 0., 2.0761e-16, 0.,
                      2.66e-18, 0., 3.e-20])
_clausen1.setflags(write=False)

_clausen2 = np.array([0., -0.96070972149008358753, 0., 4.393661151911392781e-2, 0., 7.8014905905217505e-4, 0.,
                      2.621984893260601e-5, 0., 1.09292497472610e-6, 0., 5.122618343931e-8, 0., 2.5886351267e-9, 0.,
                      1.3787545462e-10, 0., 7.63448721e-12, 0., 4.3556938e-13, 0., 2.544696e-14, 0., 1.51561e-15, 0.,
                      9.172e-17, 0., 5.63e-18, 0., 3.5e-19, 0., 2.e-20])
_clausen2.setflags(write=False)


def chebyshev(x: FloatArray, coeff: FloatArray) -> FloatArray:
    """Calculate Chebyshev series.

    :param x: Array of arguments.
    :param coeff: Chebyshev series coefficients.
    :return: Array of results.
    """
    if (x < -1.).any() or (1. < x).any():
        raise ValueError('x outside of range [-1., 1.].')
    x2 = 2. * x
    b0, b1, b2 = np.zeros(x.size), np.zeros(x.size), np.zeros(x.size)
    for i in range(len(coeff) - 1, -1, -1):
        b1, b2 = b0, b1
        b0 = x2 * b1 - b2 + coeff[i]
    return 0.5 * (b0 - b2)


def clausen(x: Union[float, FloatArray]) -> Union[float, FloatArray]:
    """Clausen integral.

    :param x: Arguments array or single value.
    :return: Array of results.
    """
    x = np.atleast_1d(np.asarray(x))
    if x.ndim != 1:
        raise ValueError('Input x must be a 1D array.')
    y = np.remainder(x + np.pi * 0.5, np.pi * 2.) - np.pi * 0.5
    res = np.zeros(x.shape)
    cond1 = (y < np.pi * 0.5) & (y != 0.)
    cond2 = (np.pi * 0.5 <= y)
    y1, y2 = y[cond1], y[cond2]
    res[cond1] = y1 - y1 * np.log(np.fabs(y1) + 1.e-20) + 0.5 * np.power(y1, 3) * chebyshev(2. * y1 / np.pi, _clausen1)
    res[cond2] = chebyshev(2. * (y2 / np.pi - 1.), _clausen2)
    return float(res) if res.size == 1 else res


def lobachevsky(x: Union[float, FloatArray]) -> Union[float, FloatArray]:
    """Milnor's Lobachevsky function.

    :param x: Arguments array or single value.
    :return: Array of results.
    """
    return clausen(2. * x) * 0.5


_MX = np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]])
_MX.setflags(write=False)


def tri_angles(lengths: FloatArray, ineq_violation: bool = True) -> FloatArray:
    """Calculate triangle angles from edge lengths. Each angle is opposite to the corresponding edge.

    :param lengths: #F by 3 edge lengths.
    :param ineq_violation: If True, triangle inequality violation leads to angles [0, 0, pi]. (CETM, Sec. 3.1).
    :return: #F by 3 triangle angles.
    """
    assert lengths.ndim == 2
    assert lengths.shape[1] == 3
    el = lengths @ _MX
    prod = np.c_[el[:, 1] * el[:, 2], el[:, 2] * el[:, 0], el[:, 0] * el[:, 1]]
    if ineq_violation:
        prod[(el < 0).any(axis=1)] = 0
    el *= np.sum(lengths, axis=1)[:, np.newaxis]
    angles = 2. * np.arctan(np.sqrt(prod / el))
    if ineq_violation:
        angles[el < 0] = np.pi
    return angles


def compute_exterior_angles_2d(points: FloatArray, ccw: bool = True) -> FloatArray:
    """Compute exterior angles of a discrete 2D curve at its vertices.

    :param points: #V by 2 curve points.
    :param ccw: Curve is counter-clockwise.
    :return: #V discrete curvatures.
    """
    edge_vec = points - np.roll(points, shift=1, axis=0)
    thetas = np.arctan2(edge_vec[:, 1], edge_vec[:, 0])
    k = np.roll(thetas, shift=-1) - thetas
    k[k < -np.pi] += 2. * np.pi
    k[np.pi < k] -= 2. * np.pi
    return k


def mobius_normalize_layout(uv: FloatArray, eps: float = 1.e-6, max_iter: int = 1000) -> FloatArray:
    """Mobius-normalize uv layout. Apex is placed at [0, 0, 1].

    :param uv: #V by 2 uv-coordinates, must have exactly one row of nan for the apex.
    :param eps: Maximal center offset error.
    :param max_iter: Maximal number of iterations. Generates a warning if reached.
    :return: #V vy 3 transformed uv on sphere.
    """
    if uv.ndim != 2:
        raise ValueError(f'uv must have array dimension 2, not {uv.ndim}.')
    if uv.shape[1] != 2:
        raise ValueError(f'uv points must be 2D, not {uv.shape[1]}D.')
    r, _ = np.where(np.isnan(uv))
    if len(r) != 2 or (apex := r[0]) != r[1]:
        raise ValueError('uv must contain exactly one row of nan where the apex is.')
    uv = uv.copy()
    uv -= np.nanmean(uv, axis=0)
    uv /= np.sqrt(np.median(np.nansum(uv ** 2, axis=1)))  # Approximate uniform distribution.
    uv[:, 0] *= -1  # Correct orientation for spherical layout.
    uv[apex] = 0.
    sumsq = np.sum(uv ** 2, axis=1)
    uv = np.c_[2. * uv, sumsq - 1.] / (sumsq + 1.)[:, np.newaxis]
    uv[apex] = [0., 0., 1.]
    # Perform Mobius normalization.
    # Algorithm 1. http://www.cs.cmu.edu/~kmcrane/Projects/MobiusRegistration/paper.pdf
    center_offset = float('inf')
    for i in range(max_iter):
        if (center_offset := np.linalg.norm(cm := uv.mean(axis=0))) < eps:
            break
        otr = np.eye(3)[np.newaxis, :, :] - np.matmul(uv[:, :, np.newaxis], uv[:, np.newaxis, :])
        neg_jac = -2. / len(uv) * otr.sum(axis=0)
        cx = np.linalg.inv(neg_jac) @ cm
        cx *= 0.5  # learning rate.
        sx = uv + cx
        uv = (1. - (cx ** 2).sum()) * sx / (sx ** 2).sum(axis=1)[:, np.newaxis] + cx
    else:
        warn(f'Premature termination. Spherical map\'s vertex center of mass is off by {center_offset}.')
    return uv


def quasi_conformal_error(source_points: npt.ArrayLike, image_points: npt.ArrayLike) -> FloatArray:
    """Compute quasi-conformal error of conformal map from source to image triangles.

    See https://hhoppe.com/tmpm.pdf

    TODO: Implement for 3D image as well.

    :param source_points: #F by 3 by 3 source points-per-triangle.
    :param image_points: #F by 3 by 2 image points-per-triangle.
    :return: #F error-per-face.
    """
    source_points, image_points = np.asarray(source_points), np.asarray(image_points)
    if source_points.ndim != 3:
        raise ValueError(f'Source points must have shape dimension 3, not {source_points.ndim}.')
    if source_points.shape[1:] != (3, 3):
        raise ValueError(f'Source points must have dimensions (#F, 3, 3) not {source_points.shape}.')
    if image_points.ndim != 3:
        raise ValueError(f'Image points must have shape dimension 3, not {image_points.ndim}.')
    if image_points.shape[1:] != (3, 2):
        raise ValueError(f'Image points must have dimensions (#F, 3, 2) not {image_points.shape}.')
    if source_points.shape[0] != image_points.shape[0]:
        raise ValueError(f'Source and image have inconsistent face counts:'
                         f' {source_points.shape[0]} != {image_points.shape[0]}.')
    uve = np.roll(image_points, shift=1, axis=1) - np.roll(image_points, shift=-1, axis=1)
    im_face_areas2 = (uve[:, 0, 0] * uve[:, 1, 1] - uve[:, 1, 0] * uve[:, 0, 1])
    jc = (uve.swapaxes(1, 2) @ source_points) / im_face_areas2[:, np.newaxis, np.newaxis]
    c, a = (jc ** 2).sum(axis=2).T
    b2 = (jc[:, 0, :] * jc[:, 1, :]).sum(axis=1) ** 2  # Note: Original expression for b is negated but squared.
    apc, sqr = a + c, np.sqrt((a - c) ** 2 + 4. * b2)
    # min_singular, max_singular = np.sqrt((apc - sqr) * 0.5), np.sqrt((apc + sqr) * 0.5)
    # error = max_singular / min_singular
    return np.sqrt((apc + sqr) / (apc - sqr))


def conformal_equivalence(points: npt.ArrayLike, faces: npt.ArrayLike,
                          image_points: npt.ArrayLike) -> Tuple[FloatArray, FloatArray]:
    """Compute conformal equivalence of conformal map from source to image triangles.

    :param points: #V by 3 source mesh points.
    :param faces: #F by 3 triangular mesh faces.
    :param image_points: #F by 3 by 2 image points-per-triangle.
    :return: #V mean evaluated log-scaling factors and #V mismatch-per-vertex (as sum of squared deviations).
    """
    points, image_points = np.asarray(points), np.asarray(image_points)
    faces = np.asarray(faces, dtype=int)
    if points.ndim != 2:
        raise ValueError(f'Points must have shape dimension 2, not {points.ndim}.')
    if points.shape[1] != 3:
        raise ValueError(f'Points must be 3-dimensional, not {points.shape[1]}D.')
    if faces.ndim != 2:
        raise ValueError(f'Faces must have shape dimension 2, not {faces.ndim}.')
    if faces.shape[1] != 3:
        raise ValueError(f'Faces must be triangles, not {faces.shape[1]}-gons.')
    if image_points.ndim != 3:
        raise ValueError(f'Image points must have shape dimension 3, not {image_points.ndim}.')
    if image_points.shape[1] != 3 or image_points.shape[2] not in (3, 2):
        raise ValueError(f'Image points must have dimensions (#F, 3, 2) or (#F, 3, 3) not {image_points.shape}.')
    if faces.shape[0] != image_points.shape[0]:
        raise ValueError(f'Source and image have inconsistent face counts:'
                         f' {faces.shape[0]} != {image_points.shape[0]}.')
    source_points = points[faces]
    elen = np.linalg.norm(np.roll(source_points, shift=1, axis=1) - np.roll(source_points, shift=-1, axis=1), axis=2)
    im_elen = np.linalg.norm(np.roll(image_points, shift=1, axis=1) - np.roll(image_points, shift=-1, axis=1), axis=2)
    ratios = im_elen / elen
    prods = np.roll(ratios, shift=-1, axis=1) * np.roll(ratios, shift=1, axis=1)
    tri_u = np.log(elen / im_elen * prods)
    vc = np.zeros(len(points), dtype=int)
    np.add.at(vc, faces, 1)
    u = np.zeros(len(points))
    np.add.at(u, faces, tri_u)
    u /= vc
    u_err = np.zeros(len(points))
    np.add.at(u_err, faces, (u[faces] - tri_u) ** 2)
    return u, u_err
