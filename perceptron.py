"""Perceptron classifier, including training and evaluation.

Input dim: 3
Output dim: 1

Ex:
Given input vector [.3, .8, -.5], predict an output between 0 and 1.

Let's suppose that if the first two features are positive and the third feature
is positive, the output should trend toward 1. We'll further say that the first
feature should provide more signal about the outcome, which can be done by
sampling with means farther from zero.
"""

import numpy as np
from sklearn.model_selection import train_test_split


class Perceptron:
    def __init__(self, alpha):
        self.weights = np.random.uniform(-1, 1, (3))
        self.bias = 1
        self.alpha = alpha

    def train(self, x_train, y_train, epochs):
        for _ in range(epochs):
            for x, y in zip(x_train, y_train):
                y_hat = self.forward(x)
                self.weights = self.weights + self.alpha * (y - y_hat) * x

    def forward(self, x):
        x_1 = np.dot(self.weights, x) + self.bias
        if x_1 >= 0:
            return 1
        else:
            return -1

    def eval(self):
        raise NotImplementedError


def main():
    # Positive samples
    n_positive = 1000
    feat_1 = np.random.normal(-.8, .3, n_positive)
    feat_2 = np.random.normal(-.2, .1, n_positive)
    feat_3 = np.random.normal(.4, .1, n_positive)
    pos_samples = np.stack((feat_1, feat_2, feat_3), axis=1)
    pos_labels = np.ones((n_positive))

    # Negative samples
    n_negative = 1000
    feat_1 = np.random.normal(.5, .3, n_negative)
    feat_2 = np.random.normal(.1, .1, n_negative)
    feat_3 = np.random.normal(.4, .1, n_negative)
    neg_samples = np.stack((feat_1, feat_2, feat_3), axis=1)
    neg_labels = np.zeros(n_positive)
    x = np.concat((pos_samples, neg_samples), axis=0)
    y = np.concat((pos_labels, neg_labels), axis=0)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, shuffle=True)

    perceptron = Perceptron(alpha=.2)
    perceptron.train(x_train, y_train, 2)
    # perceptron.eval(x_test, y_test)


if __name__ == "__main__":
    main()
