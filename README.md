# ProbNet: A Unified Probabilistic Neural Network Framework for Classification and Regression Tasks

[![GitHub release](https://img.shields.io/badge/release-0.1.0-yellow.svg)](https://github.com/thieu1995/ProbNet/releases)
[![PyPI version](https://badge.fury.io/py/probnet.svg)](https://badge.fury.io/py/probnet)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/probnet.svg)
![PyPI - Downloads](https://img.shields.io/pypi/dm/probnet.svg)
[![Downloads](https://pepy.tech/badge/probnet)](https://pepy.tech/project/probnet)
[![Tests & Publishes to PyPI](https://github.com/thieu1995/ProbNet/actions/workflows/publish-package.yaml/badge.svg)](https://github.com/thieu1995/ProbNet/actions/workflows/publish-package.yaml)
[![Documentation Status](https://readthedocs.org/projects/probnet/badge/?version=latest)](https://probnet.readthedocs.io/en/latest/?badge=latest)
[![Chat](https://img.shields.io/badge/Chat-on%20Telegram-blue)](https://t.me/+fRVCJGuGJg1mNDg1)
[![DOI](https://img.shields.io/badge/DOI-10.6084%2Fm9.figshare.28802531-blue)](https://doi.org/10.6084/m9.figshare.28802435)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)


Check out the built library such as: 

https://sklvq.readthedocs.io/en/stable/generated/sklvq.solvers.WaypointGradientDescent.html
https://www.jmlr.org/papers/volume22/21-0029/21-0029.pdf
https://github.com/MrNuggelz/sklearn-lvq  ==> They have several traditional and modern variant versions.
+ Modern variants need Gradient-descent => loss-based ==> Need Pytorch model
+ traditional variants => distance-based (rule-based) ==> Need Numpy model
===> All inherit from `BaseEstimator` class of Scikit-Learn

https://github.com/MeGysssTaa/lvq4j
+ Code in Java


https://www.mathworks.com/help/deeplearning/ref/lvqnet.html
+ Code in matlab


+ Prototype-based models in machine learning ==> This name is very general because it has many research directions:
🔹 Các phương pháp học máy dựa trên prototype:
1. Learning Vector Quantization (LVQ)
Một phương pháp học có giám sát sử dụng các prototype để phân loại dữ liệu. LVQ có nhiều biến thể như Generalized LVQ (GLVQ), Robust Soft LVQ (RSLVQ), và Relevance LVQ (GRLVQ).

2. Self-Organizing Maps (SOM)
Một mạng nơ-ron không có giám sát sử dụng các prototype để ánh xạ dữ liệu vào không gian thấp hơn, thường được sử dụng để trực quan hóa và phân cụm dữ liệu.

3. K-Means Clustering
Một thuật toán phân cụm không có giám sát sử dụng các centroid (trung tâm cụm) làm prototype để nhóm dữ liệu.

4. Gaussian Mixture Models (GMMs)
Mô hình phân phối xác suất sử dụng các thành phần Gaussian làm prototype để mô hình hóa dữ liệu phức tạp.

5. Prototypical Networks
Một phương pháp học sâu cho bài toán few-shot learning, trong đó mỗi lớp được đại diện bởi một prototype tính từ các ví dụ huấn luyện.

6. ProtoAttend
Một mô hình học sâu tích hợp cơ chế attention để xác định các prototype có liên quan, giúp cải thiện khả năng giải thích và hiệu suất mô hình.

7. Prototype Completion
Một phương pháp trong few-shot learning nhằm hoàn thiện các prototype bằng cách sử dụng kiến thức nguyên thủy hoặc thông tin bổ sung để cải thiện khả năng phân loại.

8. Prototype-Based Continual Learning
Một phương pháp học liên tục sử dụng các prototype không gắn nhãn để duy trì cấu trúc cụm trong không gian đặc trưng qua các tác vụ khác nhau.







---

## 🌟 Overview

**ProbNet** is a lightweight and extensible Python library that provides a unified implementation of 
**Probabilistic Neural Network (PNN)** and its key variant, the **General Regression Neural Network (GRNN)**. 
It supports both **classification** and **regression** tasks, making it suitable for a wide range of 
supervised learning applications.

---

## 🔧 Features

- 🧠 Full implementation of PNN for classification
- 📈 GRNN for regression modeling
- 🔍 Scikit-learn compatible interface (`fit`, `predict`, `score`)
- 🔄 Built-in support for many kernels and distance metrics
- 🧪 Fast prototyping and evaluation
- 🧩 Easily extendable and readable codebase
- 📚 Auto-generated documentation with Sphinx 
- Probabilistic models: `PnnClassifier`, `GrnnRegressor`
---

## 📦 Installation

You can install the library using `pip` (once published to PyPI):

```bash
pip install unilvq
```

After installation, you can import ProbNet as any other Python module:

```sh
$ python
>>> import unilvq
>>> unilvq.__version__
```

## 🚀 Quick Start

For Classification using PNN:

```python
from unilvq import PnnClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = PnnClassifier(sigma=0.1)
model.fit(X_train, y_train)
print("Accuracy:", model.score(X_test, y_test))
```

For Regression using GRNN:

```python
from unilvq import GrnnRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = GrnnRegressor(sigma=0.5)
model.fit(X_train, y_train)
print("R2 Score:", model.score(X_test, y_test))
```

As can be seen, you do it like any model from Scikit-Learn library such as SVC, RF, DT,...


## 📚 Documentation

Documentation is available at: 👉 https://probnet.readthedocs.io

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
We welcome contributions to `ProbNet`! If you have suggestions, improvements, or bug fixes, feel free to fork 
the repository, create a pull request, or open an issue.


## 📄 License
This project is licensed under the GPLv3 License. See the LICENSE file for more details.


## Citation Request
Please include these citations if you plan to use this library:

```bibtex
@software{thieu20250503,
  author       = {Nguyen Van Thieu},
  title        = {ProbNet: A Unified Probabilistic Neural Network Framework for Classification and Regression Tasks},
  month        = may,
  year         = 2025,
  doi         = {10.6084/m9.figshare.28802435},
  url          = {https://github.com/thieu1995/ProbNet}
}
```

## Official Links 

* Official source code repo: https://github.com/thieu1995/ProbNet
* Official document: https://probnet.readthedocs.io/
* Download releases: https://pypi.org/project/probnet/
* Issue tracker: https://github.com/thieu1995/ProbNet/issues
* Notable changes log: https://github.com/thieu1995/ProbNet/blob/master/ChangeLog.md
* Official chat group: https://t.me/+fRVCJGuGJg1mNDg1

---

Developed by: [Thieu](mailto:nguyenthieu2102@gmail.com?Subject=GrafoRVFL_QUESTIONS) @ 2025
