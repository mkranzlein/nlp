"""A self-contained feedforward neural network.

- No NumPy, no PyTorch, and matrices are just lists of lists.

There's a reason those libraries exist: performance. The goal here is not that.
It's to understand what's happening at each step, down to each matrix
multiplication, gradient calculation, and parameter update.

Some terms, with shapes in parentheses:
X (1, 5): Input matrix (1st dim 1 since we're passing one input at a time)
y (1,): Vector of binary labels, one for each input sample.
W_hidden (5, 3): Weight matrix for hidden layer
b_hidden (3,): Vector of biases for hidden layer
Z1 (1, 3): Intermediate values after affine transformation in hidden layer
A1 (1, 3): Activation values from hidden layer after ReLU applied
W_output (3, 1): Weight matrix for output layer
b_output (1,): Vector with bias for output layer
Z2 (1,): Intermediate values after affine transformation in output layer
y_hat (1,): Vector of binary model predictions for each input

L: Loss function (binary cross-entropy)

Some of the variables above could be expressed as vectors because only one
input sample is processed at a time in this network. For generalizability, these
variables are expressed as matrices with capital letters.

"""

import math

from nlp.ops import matmul, matrix_relu, relu_deriv, sigmoid, transpose


class FFNN:
    def __init__(self):
        self.input_dim = 3
        self.hidden_dim = 5
        # Initialize weights + bias for hidden layer
        self.W_hidden = None
        self.b_hidden = None

        self.output_dim = 1
        # Initialize weights + bias for output layer
        self.W_output = None
        self.b_output = None

    def forward(self, X: list):
        # --------------------------- Hidden Layer --------------------------- #
        Z1 = matmul(X, self.W_hidden)  # Shape (1, 3) x (3, 5) -> (1, 5)
        # Add bias
        for i, row in enumerate(Z1):
            for j in range(len(row)):
                row[j] += self.b_hidden[i]
        # ReLU activation
        A1 = matrix_relu(Z1)

        # --------------------------- Output Layer --------------------------- #
        Z2 = matmul(A1, self.W_output)  # Shape (1, 3) x (3, 1) -> (1, 1)
        for i, row in enumerate(Z2):
            for j in range(len(row)):
                row[j] += self.b_output[i]
        # Matrix -> vector by squeezing dim out (only 1 prediction per input)
        Z2 = Z2[0]
        # Sigmoid activation
        y_hat = sigmoid(Z2)
        return y_hat

    def loss(self, output, y):
        """Binary cross-entropy L.

        This implementation is equivalent to the BCE definition:
        $-[y * log(y_hat) + (1 - y) * log(1 - y_hat)]$

        It also follows from categorical cross-entropy (CCE) where the loss is:
        $-\\sum_{i=1}^{K}y_i * log(\\hat{y}_i$

        The BCE defintion and this implementation are a
        special case of CCE where the functionality of one-hot encoding is
        achieved by some convenient multiplications and a subtraction.
        `math.log(1 - y_hat)` conveys how confident the model is that the
        output is 0. The lower y_hat is, the better performance it has on
        samples where y is 0.
        """
        if y == 0:
            return -math.log(1 - output)
        elif y == 1:
            return -math.log(output)
        else:
            raise ValueError(f"Invalid y: {y}")

    def calc_gradient(self, X, Z1, A1, y_hat, y):
        """Calculates the gradients for each parameter in the model.

        Read any gradient, like Z2, as \\frac{\\partial{L}}{Z2}.
        `gradients["Z2"]` is "the partial derivative of the loss function
        with respect to Z2."
        """

        # Dict with partial derivatives for all weights and biases
        gradients = {}

        # Some nice math on SLP 151 means d/dZ2 L(Sigmoid(Z2)) = y_hat - y
        # This is a nice simplification that comes from the chain rule applied
        # to binary cross-entropy and then sigmoid.
        gradients["Z2"] = y_hat[0] - y[0]  # Scalar
        gradients["W_output"] = []  # (3, 1)
        for i, row in enumerate(self.W_output):
            grad_row = []
            for j, w in enumerate(row):
                # j,i because activation j,i corresponds to weight i,j
                grad_row.append(gradients["Z2"] * A1[j][i])
            gradients["W_output"].append(grad_row)

        # The multiplication by 1 isn't necessary, but it's there to show the
        # effect of the chain rule for calculating the bias gradient
        gradients["b_output"] = [gradients["Z2"] * 1 for i in range(len(self.b_output))]  # (1,)

        gradients["Z1"] = []  # (1, 3)
        for i, row in enumerate(self.W_output):
            grad_row = []
            for j, w in enumerate(row):
                grad_row.append(gradients["Z2"] * w * relu_deriv(Z1[0][i]))
            gradients["Z1"].append(grad_row)

        gradients["W_hidden"] = matmul(transpose(X), gradients["Z1"])  # (5, 3)
        gradients["b_hidden"] = [gradients["Z1"] * 1 for i in range(len(self.b_hidden))]  # (1,)

        return gradients

    def update_weights(self, gradients, lr):
        """Updates the model's weights based on gradients and learning rate."""
        for i, row in enumerate(self.W_hidden):
            for j, w in enumerate(row):
                self.W_hidden[i][j] = w - gradients["W_hidden"][i][j] * lr

        for i, b in enumerate(self.b_hidden):
            self.b_hidden[i] = b - gradients["b_hidden"][i] * lr

        for i, row in enumerate(self.W_output):
            for j, w in enumerate(row):
                self.W_output[i][j] = w - gradients["W_output"][i][j] * lr

        for i, b in enumerate(self.b_output):
            self.b_output[i] = b - gradients["b_output"][i] * lr


def train(model, x_train, y_train):
    # For each sample
    # Forward pass
    # output = model.forward()
    # gradients =
    raise NotImplementedError
