#!/usr/bin/env python
# Created by "Thieu" at 09:58, 12/05/2025 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from sklearn.datasets import load_breast_cancer
from unilvq import Lvq1Classifier, Data


## Load data object
X, y = load_breast_cancer(return_X_y=True)
data = Data(X, y)

## Split train and test
data.split_train_test(test_size=0.2, random_state=2, inplace=True)
print(data.X_train.shape, data.X_test.shape)

## Scaling dataset
data.X_train, scaler_X = data.scale(data.X_train, scaling_methods=("standard", "minmax"))
data.X_test = scaler_X.transform(data.X_test)

data.y_train, scaler_y = data.encode_label(data.y_train)
data.y_test = scaler_y.transform(data.y_test)

## Train and test
model = Lvq1Classifier(n_prototypes_per_class=1, learning_rate=0.1, seed=42)
model.fit(data.X_train, data.y_train)

## Predict
print("Predicted:", model.predict(data.X_test))

## Calculate some metrics
print(model.score(X=data.X_test, y=data.y_test))
print(model.scores(X=data.X_test, y=data.y_test, list_metrics=["PS", "RS", "NPV", "F1S", "F2S"]))
print(model.evaluate(y_true=data.y_test, y_pred=model.predict(data.X_test), list_metrics=["F2S", "CKS", "FBS"]))


#
#
# X, y = load_iris(return_X_y=True)
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
#
# model = Lvq1Classifier(n_prototypes_per_class=1, learning_rate=0.1, seed=42)
# model.fit(X_train, y_train)
# print("Accuracy (Lvq1Classifier):", model.score(X_test, y_test))
#
# model2 = Lvq21Classifier(n_prototypes_per_class=1, learning_rate=0.1, seed=42)
# model2.fit(X_train, y_train)
# print("Accuracy (LVQ2.1):", model2.score(X_test, y_test))
#
#
# Lvq3Classifier = Lvq3Classifier(n_prototypes_per_class=1, learning_rate=0.1, window=0.3, epsilon=0.2, seed=42)
# Lvq3Classifier.fit(X_train, y_train)
# print("Accuracy (Lvq3Classifier):", Lvq3Classifier.score(X_test, y_test))
#
# OptimizedLvq1Classifier = OptimizedLvq1Classifier(n_prototypes_per_class=1, initial_learning_rate=0.5, learning_decay=0.98, seed=42)
# OptimizedLvq1Classifier.fit(X_train, y_train)
# print("Accuracy (OptimizedLvq1Classifier):", OptimizedLvq1Classifier.score(X_test, y_test))
