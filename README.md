
# UniLVQ: A Unified Learning Vector Quantization Framework for Supervised Learning Tasks

[![GitHub release](https://img.shields.io/badge/release-0.1.0-yellow.svg)](https://github.com/thieu1995/UniLVQ/releases)
[![PyPI version](https://badge.fury.io/py/unilvq.svg)](https://badge.fury.io/py/unilvq)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/unilvq.svg)
![PyPI - Downloads](https://img.shields.io/pypi/dm/unilvq.svg)
[![Downloads](https://pepy.tech/badge/unilvq)](https://pepy.tech/project/unilvq)
[![Tests & Publishes to PyPI](https://github.com/thieu1995/UniLVQ/actions/workflows/publish-package.yml/badge.svg)](https://github.com/thieu1995/UniLVQ/actions/workflows/publish-package.yml)
[![Documentation Status](https://readthedocs.org/projects/unilvq/badge/?version=latest)](https://unilvq.readthedocs.io/en/latest/?badge=latest)
[![Chat](https://img.shields.io/badge/Chat-on%20Telegram-blue)](https://t.me/+fRVCJGuGJg1mNDg1)
[![DOI](https://img.shields.io/badge/DOI-10.6084%2Fm9.figshare.29002685-blue)](https://doi.org/10.6084/m9.figshare.29002685)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

---

## 📌 Overview

**UniLVQ** is an open-source Python library that provides a unified, extensible, and user-friendly 
implementation of **Learning Vector Quantization (LVQ)** algorithms for supervised learning. 
It supports both **classification** and **regression** tasks, and is designed to work seamlessly with the **scikit-learn API**.

Built on top of **NumPy** and **PyTorch**, UniLVQ combines rule-based and neural-inspired LVQ variants, 
making it suitable for both research and practical applications.


## 🚀 Features

- ✅ Unified base API compatible with `scikit-learn`
- ✅ Traditional rule-based LVQ variants: LVQ1, LVQ2.1, LVQ3, Optimized LVQ1:
  + `Lvq1Classifier`, `Lvq2Classifier`, `Lvq3Classifier`, `OptimizedLvq1Classifier`
- ✅ Loss-based LVQ models: GLVQ, GRLVQ, LGMLVQ (PyTorch-based):
  + `GlvqClassifier`, `GlvqRegressor`, `GrlvqClassifier`, `GrlvqRegressor`, `LgmlvqClassifier`
- ✅ Support for both classification and regression
- ✅ Built-in support for early stopping, metric evaluation, data scaling
- ✅ Modular design for easy extension and customization
- ✅ CI-tested, documented, and easy to use


## 🧠 Supported Models

| Type                  | Algorithms                                       | Module         |
|-----------------------|--------------------------------------------------|----------------|
| Rule-based LVQ        | LVQ1, LVQ2.1, LVQ3, Optimized LVQ1 (Classifiers) | `classic_lvq.py` |
| Generalized LVQ       | GLVQ (Classifier, Regressor)                     | `glvq.py`        |
| Generalized Relevance LVQ | GRLVQ (Classifier, Regressor)                    | `grlvq.py`       |
| Local Generalized Matrix LVQ | LGMLVQ (Classifier)                              | `lgmlvq.py`      |


## 📖 Citation Request 

Please include these citations if you plan to use this library:

```bibtex
@software{thieu20250515UniLVQ,
  author       = {Nguyen Van Thieu},
  title        = {UniLVQ: A Unified Learning Vector Quantization Framework for Supervised Learning Tasks},
  month        = June,
  year         = 2025,
  doi         = {10.6084/m9.figshare.28802435},
  url          = {https://github.com/thieu1995/UniLVQ}
}
```

## 📦 Installation

Install the latest version from PyPI:

```bash
pip install unilvq
```

Verify installation:

```sh
$ python
>>> import unilvq
>>> unilvq.__version__
```

## 🚀 Quick Start

For classification problem using LVQ1 classifier:

```python
from unilvq import Lvq1Classifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Train LVQ1 model
model = Lvq1Classifier(n_prototypes_per_class=1, learning_rate=0.1, seed=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
```

As can be seen, you do it like any model from Scikit-Learn library such as SVC, RF, DT,... Please read the [examples](/examples) folder for more use cases.


## 📚 Documentation

Documentation is available at: 👉 https://unilvq.readthedocs.io

You can build the documentation locally:

```shell
cd docs
make html
```

## 🧪 Testing
You can run unit tests using:

```shell
pytest tests/
```

## 🤝 Contributing
We welcome contributions to `UniLVQ`! If you have suggestions, improvements, or bug fixes, feel free to fork 
the repository, create a pull request, or open an issue.


## 📄 License
This project is licensed under the GPLv3 License. See the LICENSE file for more details.


## 📎 Official channels 

* 🔗 [Official source code repository](https://github.com/thieu1995/UniLVQ)
* 📘 [Official document](https://unilvq.readthedocs.io/)
* 📦 [Download releases](https://pypi.org/project/unilvq/) 
* 🐞 [Issue tracker](https://github.com/thieu1995/UniLVQ/issues) 
* 📝 [Notable changes log](/ChangeLog.md)
* 💬 [Official discussion group](https://t.me/+fRVCJGuGJg1mNDg1)

---

Developed by: [Thieu](mailto:nguyenthieu2102@gmail.com?Subject=UniLVQ_QUESTIONS) @ 2025
