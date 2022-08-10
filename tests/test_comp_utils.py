import numpy as np
from confmap.comp_utils import quantize_value, clausen, tri_angles


def test_quantize_value():
    quant = np.pi / 2.
    eps = 1.e-6
    base_series = np.arange(0, 5) * quant
    values = base_series[:, np.newaxis] + np.array([-eps, 0, eps])[np.newaxis, :]
    values = values.flatten()
    results = base_series[:, np.newaxis] + np.zeros(3)[np.newaxis, :]
    results = results.flatten()
    results[results == 0] = quant
    results[results == 2. * np.pi] = 2. * np.pi - quant
    assert np.allclose(quantize_value(values, quant), results)


def test_clausen():
    values_pi = np.arange(12) * np.pi / 12.
    results_pi = np.array([0., 0.61290614, 0.86437913, 0.98187215, 1.01494161, 0.98795895, 0.91596559, 0.80950478,
                           0.67662774, 0.52388935, 0.35690833, 0.18071658])
    values_2pi = np.concatenate([values_pi, np.pi + values_pi])
    results_2pi = np.concatenate([results_pi, [0], -np.flip(results_pi)[:-1]])
    values = np.concatenate([values_2pi - np.pi * 2, values_2pi, values_2pi + np.pi * 2])
    results = np.tile(results_2pi, 3)
    assert np.allclose(clausen(values), results)
    for value, result in zip(values, results):
        assert np.isclose(clausen(value), result)


def test_tri_angles():
    edge_lengths = np.array([[2., 2., 2.],
                             [2., 2., 3.],
                             [2., 2., 4.],
                             [2., 2., 5.]])
    results = np.array([[1.04719755, 1.04719755, 1.04719755],
                        [0.72273425, 0.72273425, 1.69612416],
                        [0., 0., 3.14159265],
                        [0., 0., 3.14159265]])
    assert np.allclose(tri_angles(edge_lengths, ineq_violation=True), results)
