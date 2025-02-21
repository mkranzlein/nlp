"""Create, train, and evaluate a feedforward neural network."""
import torch
from torch import nn

# Resources on backprop:
# - https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/
# - https://machinelearningmastery.com/the-chain-rule-of-calculus-for-univariate-and-multivariate-functions/
# - https://machinelearningmastery.com/difference-between-backpropagation-and-stochastic-gradient-descent/


class NeuralNetwork(nn.module):
    """Neural network with explicitly defined operations."""
    def __init__(self):
        super.__init__()
        self.weights = None
        self.bias = None

    def forward(x: torch.Tensor):
        """Forward pass on inputs through hidden layer."""
        raise NotImplementedError
        # Set up inputs
        # Hidden layer
        # Activation
        # Output

    def train(self, x_train, y_train, epochs: int):
        """Train model for a specified number of epochs."""
        raise NotImplementedError
        # TODO: Take slices of the input tensors as batches to avoid dataloaders
        # for transparency? Or just use dataloaders for convenience?


class SimpleNeuralNetwork(nn.module):
    """Neural network using PyTorch abstractions (e.g. `nn.Linear`)."""
    def __init__(self):
        super.__init__()


def main():
    model = NeuralNetwork()
    model.train()


if __name__ == "__main__":
    main()

# Cross Entropy
