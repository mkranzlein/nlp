"""A self-contained feedforward neural network.

- No NumPy, no PyTorch, and matrices are just lists of lists.

There's a reason those libraries exist: performance. The goal here is not that.
It's to understand what's happening at each step, down to each matrix
mulitiplication, gradient calculation, and parameter update.
"""

from nlp.ops import matmul, relu, sigmoid


class FFNN:
    def __init__(self):
        self.input_dim = 3
        self.hidden_dim = 5
        # Initialize weights + bias for hidden layer
        self.w_hidden = None
        self.b_hidden = None

        self.output_dim = 1
        # Initialize weights + bias for output layer
        self.w_out = None
        self.b_out = None

    def forward(self, x: list):
        # --------------------------- Hidden Layer --------------------------- #
        z = matmul(x, self.w_hidden)  # Shape (1, 3) x (3, 5) -> (1, 5)
        for i, row in enumerate(z):
            for j in range(len(row)):
                row[j] += self.b_hidden[i]

        # Apply ReLU element by element
        for i, row in enumerate(z):
            for j, elem in enumerate(row):
                z[i][j] = relu(elem)

        # --------------------------- Output Layer --------------------------- #
        logits = matmul(z, self.w_out)  # Shape (1, 5) x (5, 1) -> (1, 1)
        for i, row in enumerate(logits):
            for j in range(len(row)):
                row[j] += self.b_out[i]
        output = sigmoid(logits)
        return output
