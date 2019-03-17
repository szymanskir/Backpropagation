import pytest
import numpy as np

from backpropagation.network.regularizator import (
    L1Regularizator,
    L2Regularizator
)
from numpy.testing import assert_array_equal


@pytest.mark.parametrize("regularization_term, weights, n, expected", [
    (1, [np.array([[1, 2, 1]]), np.array([[1, 1, 1, 1]])], 2, 2.5),
    (1, [np.array([[1, 2], [2, 1]]), np.array([[1, 1]])], 2, 3),
])
def test_l2_regularizator_term(regularization_term, weights, n, expected):
    regularizator = L2Regularizator()
    result = regularizator.get_term(regularization_term, weights, n)

    assert result == expected


@pytest.mark.parametrize("regularization_term, weights, n, expected", [
    (1, [np.array([[1, 2, 1]]), np.array([[1, 1, 1, 1]])], 2,
     [np.array([[0.5, 1, 0.5]]), np.array([[0.5, 0.5, 0.5, 0.5]])]),
    (1, [np.array([[1, 2], [2, 1]]), np.array([[1, 1]])], 2,
     [np.array([[0.5, 1], [1, 0.5]]), np.array([[0.5, 0.5]])]),
])
def test_l2_regularizator_derivative_terms(regularization_term, weights, n, expected):
    regularizator = L2Regularizator()
    result = regularizator.get_derivative_terms(regularization_term, weights, n)

    for r, e in zip(result, expected):
        assert_array_equal(r, e)


@pytest.mark.parametrize("regularization_term, weights, n , expected", [
    (1, [np.array([[1, 2, 1]]), np.array([[1, 1, 1, 1]])], 2, 4),
    (1, [np.array([[1, 2], [2, 1]]), np.array([[1, 1]])], 2, 4),
])
def test_l1_regularizator_term(regularization_term, weights, n, expected):
    regularizator = L1Regularizator()
    result = regularizator.get_term(regularization_term, weights, n)

    assert result == expected


@pytest.mark.parametrize("regularization_term, weights, n, expected", [
    (1, [np.array([[1, 2, 1]]), np.array([[1, 1, 1, 1]])], 2,
     [np.array([[0.5, 0.5, 0.5]]), np.array([[0.5, 0.5, 0.5, 0.5]])]),
    (1, [np.array([[1, 2], [2, -1]]), np.array([[1, 1]])], 2,
     [np.array([[0.5, 0.5], [0.5, -0.5]]), np.array([[0.5, 0.5]])]),
])
def test_l1_regularizator_derivative_terms(regularization_term, weights, n, expected):
    regularizator = L1Regularizator()
    result = regularizator.get_derivative_terms(regularization_term, weights, n)

    for r, e in zip(result, expected):
        assert_array_equal(r, e)
