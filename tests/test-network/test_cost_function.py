import pytest
import numpy as np

from backpropagation.network.cost_function import (
    CrossEntropyCostFunction,
    MSECostFunction,
    SECostFunction
)
from numpy.testing import assert_array_equal


@pytest.mark.parametrize("y, a, expected", [
    (np.array([[1, 2]]), np.array([[1, 2]]), 0),
    (np.array([[1, 0]]), np.array([[0, 0]]), 0.25),
    (np.array([[1, 0, 0, 0]]), np.array([[0, 0, 0, 0]]), 0.125)
])
def test_mse_function_value(y, a, expected):
    mse_cost_function = MSECostFunction()
    result = mse_cost_function.calculate_value(y, a)

    assert_array_equal(result, expected)


@pytest.mark.parametrize("y, a, expected", [
    (np.array([[1, 2]]), np.array([[1, 2]]), np.array([[0, 0]])),
    (np.array([[1, 0]]), np.array([[1, 1]]), np.array([[0, 1]])),
])
def test_mse_function_derivative_value(y, a, expected):
    mse_cost_function = MSECostFunction()
    result = mse_cost_function.calculate_derivative_value(y, a)

    assert_array_equal(result, expected)


@pytest.mark.parametrize("y, a, expected", [
    (np.array([[1, 2]]), np.array([[1, 2]]), 0),
    (np.array([[1, 0]]), np.array([[0, 0]]), 0.5),
    (np.array([[1, 0, 0, 0]]), np.array([[0, 0, 0, 0]]), 0.5)
])
def test_se_function_value(y, a, expected):
    se_cost_function = SECostFunction()
    result = se_cost_function.calculate_value(y, a)

    assert_array_equal(result, expected)


@pytest.mark.parametrize("y, a, expected", [
    (np.array([[1, 2]]), np.array([[1, 2]]), np.array([[0, 0]])),
    (np.array([[1, 0]]), np.array([[1, 1]]), np.array([[0, 1]])),
])
def test_se_function_derivative_value(y, a, expected):
    se_cost_function = SECostFunction()
    result = se_cost_function.calculate_derivative_value(y, a)

    assert_array_equal(result, expected)
