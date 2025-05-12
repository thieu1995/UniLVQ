#!/usr/bin/env python
# Created by "Thieu" at 10:02, 12/05/2025 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from sklearn.datasets import load_iris
from unilvq import Lvq1Classifier, Data


## Load data object
X, y = load_iris(return_X_y=True)
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
