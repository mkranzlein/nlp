import torch
from torch import nn


class NeuralNetwork(nn.module):
    """Neural network with explicitly defined operations."""
    def __init__(self):
        super.__init__()
        self.weights = None
        self.bias = None

    def forward(x):
        raise NotImplementedError
        # Set up inputs
        # Hidden layer
        # Activation
        # Output

    def train(self):
        raise NotImplementedError


class SimpleNeuralNetwork(nn.module):
    """Neural network using PyTorch abstractions (e.g. `nn.Linear`)."""
    def __init__(self):
        super.__init__()


def main():
    model = NeuralNetwork()
    model.train()


if __name__ == "__main__":
    main()
