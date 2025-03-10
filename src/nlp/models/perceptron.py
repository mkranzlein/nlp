"""Binary classification using a single perceptron.

Given an input vector with 3 features (e.g. [.3, .8, -.5]), predict an output
of 0 or 1.

For this implementation, features 1 and 2 are inversely correlated with the
output and feature three is directly correlated with the output. A dataset is
generated drawing samples from normal distributions centered around positive
or negative means as appropriate.

Through training, the perceptron should learn negative weights for features 1
and 2 and a positive weight for feature 3.
"""

import numpy as np
from sklearn.model_selection import train_test_split


class Perceptron:
    def __init__(self, alpha):
        self.weights = np.random.uniform(-1, 1, (3))
        self.bias = 1
        self.alpha = alpha

    def forward(self, x):
        x_1 = np.dot(self.weights, x) + self.bias
        if x_1 >= 0:
            return 1
        else:
            return -1


def train(model, x_train, y_train, epochs):
    for _ in range(epochs):
        for x, y in zip(x_train, y_train):
            y_hat = model.forward(x)
            model.weights = model.weights + model.alpha * (y - y_hat) * x


def eval(self):
    raise NotImplementedError


def generate_dataset(n_positive, n_negative):
    # Positive samples
    feat_1 = np.random.normal(-.8, .3, n_positive)
    feat_2 = np.random.normal(-.2, .1, n_positive)
    feat_3 = np.random.normal(.4, .1, n_positive)
    pos_samples = np.stack((feat_1, feat_2, feat_3), axis=1)
    pos_labels = np.ones((n_positive))

    # Negative samples
    feat_1 = np.random.normal(.5, .3, n_negative)
    feat_2 = np.random.normal(.1, .1, n_negative)
    feat_3 = np.random.normal(.4, .1, n_negative)
    neg_samples = np.stack((feat_1, feat_2, feat_3), axis=1)
    neg_labels = np.zeros(n_positive)
    x = np.concat((pos_samples, neg_samples), axis=0)
    y = np.concat((pos_labels, neg_labels), axis=0)
    return x, y


def main():
    x, y = generate_dataset(n_positive=1000, n_negative=1000)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, shuffle=True)

    perceptron = Perceptron(alpha=.2)
    perceptron.train(x_train, y_train, 2)
    # perceptron.eval(x_test, y_test)


if __name__ == "__main__":
    main()
