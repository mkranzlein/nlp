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
import random

from nlp.ops import matmul, matrix_relu, relu_deriv, sigmoid, transpose


class FFNN:
    def __init__(self):
        
        self.input_dim = 5
        self.hidden_dim = 3
        # Initialize weights + bias for hidden layer
        self.W_hidden = [[random.normalvariate(0, .3) for j in range(self.hidden_dim)]
                         for i in range(self.input_dim)]
        self.b_hidden = [random.normalvariate(0, .3) for i in range(self.hidden_dim)]

        self.output_dim = 1
        # Initialize weights + bias for output layer
        self.W_output = [[random.normalvariate(0, .3) for j in range(self.output_dim)]
                         for i in range(self.hidden_dim)]
        self.b_output = [random.normalvariate(0, .3)]

    def forward(self, X: list):
        # --------------------------- Hidden Layer --------------------------- #
        Z1 = matmul(X, self.W_hidden)  # Shape (1, 5) x (5, 3) -> (1, 3)
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
        # Doing some hacky indexing here. Sigmoid takes a float, but we treat Z2
        # as a 1-dim vector and want y_hat to be 1-dim vector as well (ideally).
        # In theory, these vectors would make it easier to do batched forward
        # passes, but then I guess sigmoid would need to accept vector inputs.
        y_hat = sigmoid(Z2[0])
        return y_hat, A1, Z1

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

    def calc_gradients(self, X, Z1, A1, y_hat, y):
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

        gradients["W_hidden"] = matmul(transpose(X), transpose(gradients["Z1"]))  # (5, 3)
        gradients["b_hidden"] = [gradients["Z1"][i][0] * 1 for i in range(len(self.b_hidden))]  # (1,)

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


def generate_dataset(n_positive, n_negative):
    num_features = 5
    means = [-.5, .8, .4, .2, -.9]  # Where to center normal dist for each feature
    pos_features = []
    neg_features = []
    for i in range(num_features):
        pos_features.append([random.normalvariate(mu=means[i], sigma=.1)
                             for n in range(n_positive)])
        neg_features.append([random.normalvariate(mu=means[i] * -1, sigma=.1)
                             for n in range(n_negative)])

    X = transpose(pos_features)
    X.extend(transpose(neg_features))

    y = [1] * n_positive + [0] * n_negative
    return X, y


def train(model: FFNN, X, y, epochs: int = 3):
    for epoch in range(epochs):
        for input_sample, label in zip(X, y):
            y_hat, A1, Z1 = model.forward([input_sample])
            gradients = model.calc_gradients([input_sample],
                                            Z1,
                                            A1,
                                            [y_hat],
                                            [label])
            model.update_weights(gradients, .001)
        print(f"Epoch {epoch} completed:", end=" ")
        evaluate(model, X, y)


def evaluate(model, X, y):
    num_correct = 0
    for x, label in zip(X, y):
        y_hat, _, _ = model.forward([x])
        if y_hat > .5:
            pred = 1
        else:
            pred = 0
        if pred == label:
            num_correct += 1
    print(f"{num_correct} correct out of {len(X)}")

