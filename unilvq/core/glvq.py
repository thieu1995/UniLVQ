#!/usr/bin/env python
# Created by "Thieu" at 02:19, 12/05/2025 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from unilvq.common.early_stopper import EarlyStopper
from unilvq.common.misc import BaseModel


class CustomGLVQ(nn.Module):
    """
    Custom implementation of the Generalized Learning Vector Quantization (GLVQ) model.

    This class defines a neural network-based GLVQ model with trainable prototypes and
    a custom loss function for classification tasks.

    Attributes
    ----------
    prototypes : torch.nn.Parameter
        A tensor containing the trainable prototypes of shape (n_prototypes, input_dim).
    prototype_labels : torch.nn.Parameter
        A tensor containing the labels of the prototypes, with shape (n_prototypes,).

    Methods
    -------
    forward(x):
        Computes the squared Euclidean distance between input samples and prototypes.
    glvq_loss(dists, y_true):
        Computes the GLVQ loss based on distances and true labels.
    """
    def __init__(self, input_dim, n_prototypes, n_classes, seed=None):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        self.prototypes = nn.Parameter(torch.randn(n_prototypes, input_dim))
        self.prototype_labels = nn.Parameter(torch.randint(0, n_classes, (n_prototypes,)), requires_grad=False)

    def forward(self, x):
        """
        Compute the squared Euclidean distance between input samples and prototypes.
        """
        return torch.cdist(x, self.prototypes, p=2) ** 2

    def glvq_loss(self, dists, y_true):
        """
        Compute the GLVQ loss based on distances and true labels.
        """
        batch_size = dists.shape[0]
        y_true = y_true.view(-1, 1)
        proto_labels = self.prototype_labels.expand(batch_size, -1)
        true_mask = (proto_labels == y_true)
        false_mask = ~true_mask

        d_plus = torch.min(dists.masked_fill(~true_mask, float('inf')), dim=1).values
        d_minus = torch.min(dists.masked_fill(~false_mask, float('inf')), dim=1).values

        mu = (d_plus - d_minus) / (d_plus + d_minus + 1e-8)
        return torch.mean(torch.sigmoid(mu))

