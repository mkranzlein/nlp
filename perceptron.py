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

import torch

# ---------------------------------------------------------------------------- #
#                                 Generate data                                #
# ---------------------------------------------------------------------------- #

# Positive samples
n_pos = 100
feat_1= torch.normal(mean=-0.8, std=.3, size=(n_pos, 1))
feat_2 = torch.normal(mean=-.2, std=.1, size=(n_pos, 1))
feat_3 = torch.normal(mean=.4, std=.1, size=(n_pos, 1))

pos_samples = torch.cat((feat_1, feat_2, feat_3), dim=1)
pos_labels = torch.ones((100, 1))

# Negative samples
n_neg = 100
feat_1= torch.normal(mean=.5, std=.3, size=(n_pos, 1))
feat_2 = torch.normal(mean=.1, std=.1, size=(n_pos, 1))
feat_3 = torch.normal(mean=.4, std=.1, size=(n_pos, 1))
neg_samples = torch.cat((feat_1, feat_2, feat_3), dim=1)
neg_labels = torch.zeros((100, 1))
x = torch.cat(pos_samples, neg_samples, dim=0)
y = torch.cat(pos_labels, neg_labels, dim=0)

# TODOs:
# Shuffle x and y together
# Initialize weights
# Forward pass
# Training loop
# Eval
