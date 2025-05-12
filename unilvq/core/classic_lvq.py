#!/usr/bin/env python
# Created by "Thieu" at 02:18, 12/05/2025 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import inspect
import pickle
import pprint
from pathlib import Path
import pandas as pd
import numpy as np
from permetrics import ClassificationMetric
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.utils import check_random_state


class BaseLVQ(BaseEstimator):
    def __init__(self, n_prototypes_per_class=1, learning_rate=0.1, seed=None):
        self.n_prototypes_per_class = n_prototypes_per_class
        self.learning_rate = learning_rate
        self.seed = seed

    def _initialize_prototypes(self, X, y):
        rng = check_random_state(self.seed)
        self.classes_ = np.unique(y)
        self.n_classes = len(self.classes_)
        self.prototypes_ = []
        self.prototype_labels_ = []

        for cls in self.classes_:
            idx = np.where(y == cls)[0]
            selected = rng.choice(idx, self.n_prototypes_per_class, replace=False)
            self.prototypes_.append(X[selected])
            self.prototype_labels_ += [cls] * self.n_prototypes_per_class

        self.prototypes_ = np.vstack(self.prototypes_)
        self.prototype_labels_ = np.array(self.prototype_labels_)

    def _predict_sample(self, x):
        dists = np.linalg.norm(self.prototypes_ - x, axis=1)
        winner = np.argmin(dists)
        return self.prototype_labels_[winner]

    def predict(self, X):
        return np.array([self._predict_sample(x) for x in X])

    def evaluate(self, y_true, y_pred, list_metrics=("AS", "RS")):
        """Return the list of classification performance metrics of the prediction.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True values for `X`.

        y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Predicted values for `X`.

        list_metrics : list
            You can get classification metrics from Permetrics library: https://permetrics.readthedocs.io/en/latest/pages/classification.html

        Returns
        -------
        results : dict
            The results of the list metrics
        """
        cm = ClassificationMetric(y_true, y_pred)
        return cm.get_metrics_by_list_names(list_metrics)

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))

    def scores(self, X, y, list_metrics=("AS", "RS")):
        """Return the list of classification metrics of the prediction.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
           Test samples. For some estimators this may be a precomputed kernel matrix or a list of generic objects instead with shape
           ``(n_samples, n_samples_fitted)``, where ``n_samples_fitted`` is the number of samples used in the fitting for the estimator.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
           True values for `X`.

        list_metrics : list, default=("AS", "RS")
           You can get classification metrics from Permetrics library: https://permetrics.readthedocs.io/en/latest/pages/classification.html

        Returns
        -------
        results : dict
           The results of the list metrics
        """
        y_pred = self.predict(X)
        return self.evaluate(y, y_pred, list_metrics)

    def __repr__(self, **kwargs):
        """Pretty-print parameters like scikit-learn's Estimator.
        """
        param_order = list(inspect.signature(self.__init__).parameters.keys())
        param_dict = {k: getattr(self, k) for k in param_order}

        param_str = ", ".join(f"{k}={repr(v)}" for k, v in param_dict.items())
        if len(param_str) <= 80:
            return f"{self.__class__.__name__}({param_str})"
        else:
            formatted_params = ",\n  ".join(f"{k}={pprint.pformat(v)}" for k, v in param_dict.items())
            return f"{self.__class__.__name__}(\n  {formatted_params}\n)"

    def save_metrics(self, y_true, y_pred, list_metrics=("RMSE", "MAE"), save_path="history", filename="metrics.csv"):
        """
        Save evaluation metrics to csv file

        Parameters
        ----------
        y_true : ground truth data
        y_pred : predicted output
        list_metrics : list of evaluation metrics
        save_path : saved path (relative path, consider from current executed script path)
        filename : name of the file, needs to have ".csv" extension
        """
        Path(save_path).mkdir(parents=True, exist_ok=True)
        results = self.evaluate(y_true, y_pred, list_metrics)
        df = pd.DataFrame.from_dict(results, orient='index').T
        df.to_csv(f"{save_path}/{filename}", index=False)

    def save_y_predicted(self, X, y_true, save_path="history", filename="y_predicted.csv"):
        """
        Save the predicted results to csv file

        Parameters
        ----------
        X : The features data, nd.ndarray
        y_true : The ground truth data
        save_path : saved path (relative path, consider from current executed script path)
        filename : name of the file, needs to have ".csv" extension
        """
        Path(save_path).mkdir(parents=True, exist_ok=True)
        y_pred = self.predict(X)
        data = {"y_true": np.squeeze(np.asarray(y_true)), "y_pred": np.squeeze(np.asarray(y_pred))}
        pd.DataFrame(data).to_csv(f"{save_path}/{filename}", index=False)

    def save_model(self, save_path="history", filename="model.pkl"):
        """
        Save model to pickle file

        Parameters
        ----------
        save_path : saved path (relative path, consider from current executed script path)
        filename : name of the file, needs to have ".pkl" extension
        """
        Path(save_path).mkdir(parents=True, exist_ok=True)
        if filename[-4:] != ".pkl":
            filename += ".pkl"
        pickle.dump(self, open(f"{save_path}/{filename}", 'wb'))

    @staticmethod
    def load_model(load_path="history", filename="model.pkl"):
        """
        Parameters
        ----------
        load_path : str, optional
            Directory path where the model file is located. Defaults to "history".
        filename : str
            Name of the file to be loaded. If the filename doesn't end with ".pkl", the extension is automatically added.

        Returns
        -------
        object
            The model loaded from the specified pickle file.
        """
        if filename[-4:] != ".pkl":
            filename += ".pkl"
        return pickle.load(open(f"{load_path}/{filename}", 'rb'))


class Lvq1Classifier(BaseLVQ, ClassifierMixin):

    def __init__(self, n_prototypes_per_class=1, learning_rate=0.1, seed=None):
        super().__init__(n_prototypes_per_class, learning_rate, seed)

    def fit(self, X, y):
        self._initialize_prototypes(X, y)
        for xi, yi in zip(X, y):
            dists = np.linalg.norm(self.prototypes_ - xi, axis=1)
            winner_idx = np.argmin(dists)
            winner_label = self.prototype_labels_[winner_idx]

            if winner_label == yi:
                self.prototypes_[winner_idx] += self.learning_rate * (xi - self.prototypes_[winner_idx])
            else:
                self.prototypes_[winner_idx] -= self.learning_rate * (xi - self.prototypes_[winner_idx])
        return self


class Lvq21Classifier(BaseLVQ, ClassifierMixin):
    def __init__(self, n_prototypes_per_class=1, learning_rate=0.1, window=0.3, seed=None):
        super().__init__(n_prototypes_per_class, learning_rate, seed)
        self.window = window

    def fit(self, X, y):
        self._initialize_prototypes(X, y)

        for xi, yi in zip(X, y):
            dists = np.linalg.norm(self.prototypes_ - xi, axis=1)
            idx_sorted = np.argsort(dists)
            i, j = idx_sorted[0], idx_sorted[1]

            label_i, label_j = self.prototype_labels_[i], self.prototype_labels_[j]

            # Check if window condition satisfied
            dist_ratio = dists[i] / dists[j]
            if (label_i != label_j and
                    ((label_i == yi and label_j != yi) or (label_i != yi and label_j == yi)) and
                    ((1 - self.window) <= dist_ratio <= (1 + self.window))):

                if label_i == yi:
                    self.prototypes_[i] += self.learning_rate * (xi - self.prototypes_[i])
                    self.prototypes_[j] -= self.learning_rate * (xi - self.prototypes_[j])
                else:
                    self.prototypes_[i] -= self.learning_rate * (xi - self.prototypes_[i])
                    self.prototypes_[j] += self.learning_rate * (xi - self.prototypes_[j])
        return self


class Lvq3Classifier(BaseLVQ, ClassifierMixin):
    def __init__(self, n_prototypes_per_class=1, learning_rate=0.1, window=0.3, epsilon=0.3, seed=None):
        super().__init__(n_prototypes_per_class, learning_rate, seed)
        self.window = window
        self.epsilon = epsilon

    def fit(self, X, y):
        self._initialize_prototypes(X, y)

        for xi, yi in zip(X, y):
            dists = np.linalg.norm(self.prototypes_ - xi, axis=1)
            idx_sorted = np.argsort(dists)
            i, j = idx_sorted[0], idx_sorted[1]

            label_i, label_j = self.prototype_labels_[i], self.prototype_labels_[j]
            dist_ratio = dists[i] / dists[j]

            if label_i != label_j and ((1 - self.window) <= dist_ratio <= (1 + self.window)):
                # Nếu cả hai prototype đều khác class với yi thì bỏ qua
                if label_i == yi and label_j != yi:
                    self.prototypes_[i] += self.learning_rate * (xi - self.prototypes_[i])
                    self.prototypes_[j] -= self.epsilon * self.learning_rate * (xi - self.prototypes_[j])
                elif label_j == yi and label_i != yi:
                    self.prototypes_[j] += self.learning_rate * (xi - self.prototypes_[j])
                    self.prototypes_[i] -= self.epsilon * self.learning_rate * (xi - self.prototypes_[i])
                elif label_i != yi and label_j != yi:
                    # No update, both are incorrect
                    continue
        return self

