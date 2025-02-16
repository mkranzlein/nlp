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


model = NeuralNetwork()
model.train()


class SimpleNeuralNetwork(nn.module):
    """Neural network using PyTorch abstractions (e.g. `nn.Linear`)."""
    def __init__(self):
        super.__init__()
