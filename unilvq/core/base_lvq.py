#!/usr/bin/env python
# Created by "Thieu" at 10:05, 11/05/2025 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin



class GRLVQ(nn.Module):
    def __init__(self, input_dim, num_prototypes, output_dim=1):
        """
        Generalized Relevance LVQ for regression or classification.

        Args:
            input_dim (int): Dimension of input features
            num_prototypes (int): Number of prototype vectors
            output_dim (int): Output dimension (1 for regression, >1 for classification one-hot)
        """
        super(GRLVQ, self).__init__()
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, input_dim))
        self.prototype_outputs = nn.Parameter(torch.randn(num_prototypes, output_dim))  # Can be real-valued
        self.relevance = nn.Parameter(torch.ones(input_dim))  # Diagonal relevance matrix

    def forward(self, x):
        # Apply relevance weights (Mahalanobis-like distance)
        relevance_weights = self.relevance**2  # ensure positivity
        x = x.unsqueeze(1)  # (B, 1, D)
        proto = self.prototypes.unsqueeze(0)  # (1, P, D)
        dist = ((x - proto) ** 2 * relevance_weights).sum(dim=2)  # (B, P)
        idx = torch.argmin(dist, dim=1)  # (B,)
        output = self.prototype_outputs[idx]  # (B, output_dim)
        return output

class GRLVQRegressor(BaseEstimator):
    def __init__(self, input_dim, num_prototypes=10, epochs=100, lr=0.01, verbose=False):
        self.input_dim = input_dim
        self.num_prototypes = num_prototypes
        self.epochs = epochs
        self.lr = lr
        self.verbose = verbose

    def fit(self, X, y):
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        self.model = GRLVQ(input_dim=self.input_dim,
                           num_prototypes=self.num_prototypes,
                           output_dim=1)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        for epoch in range(self.epochs):
            self.model.train()
            y_pred = self.model(X)
            loss = F.mse_loss(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if self.verbose and (epoch+1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.epochs}, Loss: {loss.item():.4f}")
        return self

    def predict(self, X):
        X = torch.tensor(X, dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(X)
        return y_pred.numpy().flatten()


from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_diabetes
from unilvq import Data
from permetrics import RegressionMetric

## Load data object
X, y = load_diabetes(return_X_y=True)
data = Data(X, y)

## Split train and test
data.split_train_test(test_size=0.2, random_state=42, inplace=True)
print(data.X_train.shape, data.X_test.shape)

## Scaling dataset
data.X_train, scaler_X = data.scale(data.X_train, scaling_methods=("standard", "minmax"))
data.X_test = scaler_X.transform(data.X_test)

data.y_train, scaler_y = data.scale(data.y_train, scaling_methods=("standard", "minmax"))
data.y_train = data.y_train.ravel()
data.y_test = scaler_y.transform(data.y_test.reshape(-1, 1)).ravel()

# X, y = load_diabetes(return_X_y=True)
#
# X, y = make_regression(n_samples=500, n_features=5, noise=10)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

reg = GRLVQRegressor(input_dim=10, num_prototypes=15, epochs=1000, lr=0.5, verbose=True)
reg.fit(data.X_train, data.y_train)
y_pred = reg.predict(data.X_test)

mt = RegressionMetric(data.y_test, y_pred)
print(mt.get_metrics_by_list_names(["RMSE", "R", "KGE", "NNSE"]))

print("MSE:", mean_squared_error(data.y_test, y_pred))
