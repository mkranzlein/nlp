"""A self-contained feedforward neural network.

No NumPy, no PyTorch, and matrices are just lists of lists.

NumPy and PyTorch have better performance and add quality of life features.
The goal here is to understand what's happening at each step, down to each
matrix multiplication, gradient calculation, and parameter update.

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

Some variables like X could be a vector instead of a matrix, seeing as this
implementation was primarily intended to compute forward passes on single input
samples. However, I implemented most of this network with an eye toward being
able to accept matrix inputs. This is why there are several matrices that have
a dim equal to 1.

I try to express matrices as capital letters (e.g `X`) and vectors as
lowercase (e.g. `y`).
"""

import math
import random

from nlp.ops import matmul, matrix_relu, relu_deriv, sigmoid, transpose


class FFNN:
    """Feed-forward Neural network.

    - 5 input features
    - 1 hidden layer with dim 3
    - 1 output
    - ReLU applied at hidden layer; sigmoid applied at output
    """
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

    def forward(self, X: list[list[float]]) -> tuple[float,
                                                     list[list[float]],
                                                     list[float]]:
        """Forward pass through the network.

        Args:
            X: An input matrix (1, hidden_dim).

        Returns:
            (y_hat, A1, Z1)
        """
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

    def loss(self, y_hat: float, y: int) -> float:
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

        Args:
            y_hat: Model output. Scalar.
            y: Ground truth label for an input sample. Scalar.

        Returns:
            A scalar loss value.
        """
        if y == 0:
            return -math.log(1 - y_hat)
        elif y == 1:
            return -math.log(y_hat)
        else:
            raise ValueError(f"Invalid y: {y}")

    def calc_gradients(self, X: list[list],
                       Z1: list[list[float]], A1: list[list[float]],
                       y_hat: list[float], y: list[int]) -> dict:
        """Calculates the gradients for each parameter in the model.

        Read any gradient, like Z2, as \\frac{\\partial{L}}{Z2}.
        `gradients["Z2"]` is "the partial derivative of the loss function
        with respect to Z2."

        Args:
            X: Input matrix (generally a minibatch, here just 1 sample).
            Z1: Intermediate result W_hidden * X + b_hidden.
            A1: Neuron activation ReLU(Z1).
            y_hat: Sigmoided model logits.
            y: Ground truth label(s) for X.
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

    def update_weights(self, gradients: dict, lr: float) -> None:
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


def generate_dataset(n_pos: int, n_neg: int) -> tuple[list[list[float]],
                                                      list[float]]:
    """Generates a dataset of size n, equal to n_pos + n_neg.

    Items in the dataset are 5-dim feature vectors sampled from 5 normal
    distributions with arbitrary preset means. Negative samples are generated
    by drawing from distributions with those same means inverted. This is to
    ensure there's some pattern in the data that the network can learn.

    Returns:
        X: A matrix of input features with shape (n, num_features)
        y: A vector of binary labels with shape (n,)
    """
    num_features = 5
    means = [-.5, .8, .4, .2, -.9]  # Where to center each normal dist
    pos_features = []
    neg_features = []
    for i in range(num_features):
        pos_features.append([random.normalvariate(mu=means[i], sigma=.1)
                             for n in range(n_pos)])
        neg_features.append([random.normalvariate(mu=means[i] * -1, sigma=.1)
                             for n in range(n_neg)])

    X = transpose(pos_features)
    X.extend(transpose(neg_features))

    y = [1] * n_pos + [0] * n_neg
    return X, y


def train(model: FFNN, X: list[list[float]], y: list[float],
          epochs: int = 3) -> None:
    """Trains the feed-forward network for a specified number of epochs."""
    for epoch in range(epochs):
        for input_sample, label in zip(X, y):
            y_hat, A1, Z1 = model.forward([input_sample])
            gradients = model.calc_gradients([input_sample], Z1, A1,
                                             [y_hat], [label])
            model.update_weights(gradients, .001)
        print(f"Epoch {epoch} completed:", end=" ")
        evaluate(model, X, y)


def evaluate(model: FFNN, X: list[list[float]], y: list[float]):
    """Calculates model accuracy on a labeled dataset."""
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


def main():
    random.seed(0)
    X, y = generate_dataset(100, 100)
    model = FFNN()
    print("Initial accuracy: ", end=" ")
    evaluate(model, X, y)
    train(model, X, y, epochs=6)


if __name__ == "__main__":
    main()
