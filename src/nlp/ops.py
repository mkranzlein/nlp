"""Math operations for ML from scratch."""

import math


def softmax(z: list):
    """Performs softmax on a vector z."""
    denominator = sum([math.exp(z_j) for z_j in z])
    result = [math.exp(z_i) / denominator for z_i in z]
    return result


def sigmoid(z: float):
    """Sigmoid—Squishes a value into the range (0,1)."""
    return 1 / (1 + math.exp(-z))


def relu(val):
    """Identity function for inputs >= 0. 0 for negative inputs."""
    return max(val, 0)


def get_matrix_shape(x: list[list]):
    """Returns the shape of a 2-D matrix represented as a list of lists.

    Returns -1 if shape is invalid (e.g. rows aren't all the same len).
    """
    dim_1 = len(x)
    dim_2 = len(x[0])
    for row in x:
        n_cols = len(row)
        if n_cols != dim_2:
            return -1
    return (dim_1, dim_2)


def dot_prod(vec_1: list, vec_2: list):
    """Dot product of two vectors."""
    assert len(vec_1) == len(vec_2)
    result = 0
    for i in range(len(vec_1)):
        result += vec_1[i] * vec_2[i]
    return result


def matmul(a: list[list], b: list[list]):
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
