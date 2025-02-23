"""Math operations for ML from scratch."""

import copy
import math


def softmax(z: list[float]) -> list[float]:
    """Performs softmax on a vector z."""
    denominator = sum([math.exp(z_j) for z_j in z])
    result = [math.exp(z_i) / denominator for z_i in z]
    return result


def sigmoid(z: float) -> float:
    """Sigmoid—Squishes a value into the range (0,1)."""
    return 1 / (1 + math.exp(-z))


def relu(z: float) -> int:
    """Identity function for inputs >= 0. 0 for negative inputs."""
    return max(z, 0)


def relu_deriv(z: float) -> int:
    """First derivative of ReLU."""
    if z >= 0:
        return 1
    else:
        return 0


def matrix_relu(Z: list[list[float]]) -> list[list[int]]:
    """Applies ReLU element by element, returning a new matrix."""
    result = copy.deepcopy(Z)
    for i, row in enumerate(result):
        for j, elem in enumerate(row):
            result[i][j] = relu(elem)
    return result


def get_matrix_shape(X: list[list]) -> tuple[int, int]:
    """Returns the shape of a 2-D matrix represented as a list of lists.

    Returns -1 if shape is invalid (e.g. rows aren't all the same len).
    """
    dim_1 = len(X)
    dim_2 = len(X[0])
    for row in X:
        n_cols = len(row)
        if n_cols != dim_2:
            return -1
    return (dim_1, dim_2)


def transpose(X: list[list]) -> list[list]:
    """Returns the transpose of a matrix X.

    m x n -> n x m
    """
    result = []
    m = len(X)
    n = len(X[0])
    for j in range(n):
        new_row = [X[i][j] for i in range(m)]
        result.append(new_row)
    return result


def dot_prod(vec_1: list[float], vec_2: list[float]) -> list[float]:
    """Dot product of two vectors."""
    assert len(vec_1) == len(vec_2)
    result = 0
    for i in range(len(vec_1)):
        result += vec_1[i] * vec_2[i]
    return result


def matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    """Multiply two 2-D matrices."""
    # Verify shapes are compatible
    a_shape = get_matrix_shape(a)
    assert a_shape != -1
    b_shape = get_matrix_shape(b)
    assert b_shape != -1
    # Input matrices with dims m x n and n x p yield m x p matrix
    if a_shape[1] != b_shape[0]:
        raise ValueError(f"Incompatible shapes: {a_shape} and {b_shape}.")

    result = []
    # Position i,j in the result matrix is the dot product of the ith row of a
    # and the jth column of matrix b
    for i in range(a_shape[0]):
        result_row = []
        for j in range(b_shape[1]):
            row_i = a[i]

            # Column j is the jth element of each row
            col_j = [row[j] for row in b]
            result_row.append(dot_prod(row_i, col_j))
        result.append(result_row)
    return result
