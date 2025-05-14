#!/usr/bin/env python
# Created by "Thieu" at 02:20, 12/05/2025 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from unilvq import EarlyStopper
from unilvq.core.base_model import BaseModel


class CustomGRLVQ(nn.Module):
    """
    Custom implementation of the Generalized Relevance Learning Vector Quantization (GRLVQ) model.

    This class defines a neural network-based GRLVQ model with trainable prototypes,
    relevance matrices, and a custom loss function.

    Attributes
    ----------
    input_dim : int
        The dimensionality of the input data.
    n_prototypes : int
        The number of prototypes used in the model.
    prototypes : torch.nn.Parameter
        A tensor containing the trainable prototypes of shape (n_prototypes, input_dim).
    prototype_labels : torch.nn.Parameter
        A tensor containing the labels of the prototypes, with shape (n_prototypes,).
    relevance : torch.nn.Parameter
        A tensor representing the relevance matrix or vector, depending on the relevance type.
    relevance_type : str
        The type of relevance used ('diag' for diagonal relevance or 'matrix' for full matrix relevance).

    Methods
    -------
    _diag_distance(x):
        Computes the squared Euclidean distance with diagonal relevance weighting.
    _matrix_distance(x):
        Computes the squared Euclidean distance with full matrix relevance weighting.
    forward(x):
        Computes the distances between input samples and prototypes using the specified relevance type.
    grlvq_loss(dists, y_true):
        Computes the GRLVQ loss based on distances and true labels.
    """

    def __init__(self, input_dim, n_prototypes, n_classes, relevance_type='diag', seed=None):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        
        self.input_dim = input_dim
        self.n_prototypes = n_prototypes
        self.prototypes = nn.Parameter(torch.randn(n_prototypes, input_dim))
        self.prototype_labels = nn.Parameter(torch.randint(0, n_classes, (n_prototypes,)), requires_grad=False)
        
        if relevance_type == 'diag':
            self.relevance = nn.Parameter(torch.ones(input_dim))
            self.distance = self._diag_distance
        elif relevance_type == 'matrix':
            self.relevance = nn.Parameter(torch.eye(input_dim))  # Full matrix
            self.distance = self._matrix_distance
        else:
            raise ValueError("relevance_type must be 'diag' or 'matrix'")
    
    def _diag_distance(self, x):
        """
        Computes the squared Euclidean distance with diagonal relevance weighting.
        """
        w = self.relevance.abs()  # enforce positive relevance
        dists = ((x.unsqueeze(1) - self.prototypes.unsqueeze(0)) ** 2) * w
        return dists.sum(dim=2)
    
    def _matrix_distance(self, x):
        """
        Computes the squared Euclidean distance with full matrix relevance weighting.
        """
        W = self.relevance @ self.relevance.T  # positive semi-definite
        diffs = x.unsqueeze(1) - self.prototypes.unsqueeze(0)
        dists = torch.einsum('bij,jk,bik->bi', diffs, W, diffs)
        return dists

    def forward(self, x):
        """
        Computes the distances between input samples and prototypes using the specified relevance type.
        """
        return self.distance(x)

    def grlvq_loss(self, dists, y_true):
        """
        Computes the GRLVQ loss based on distances and true labels.
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

