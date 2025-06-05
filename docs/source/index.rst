.. UniLVQ documentation master file, created by
   sphinx-quickstart on Sat May 20 16:59:33 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to UniLVQ's documentation!
==================================

.. image:: https://img.shields.io/badge/release-0.1.0-yellow.svg
   :target: https://github.com/thieu1995/UniLVQ/releases

.. image:: https://badge.fury.io/py/unilvq.svg
   :target: https://badge.fury.io/py/unilvq

.. image:: https://img.shields.io/pypi/pyversions/unilvq.svg
   :target: https://www.python.org/

.. image:: https://img.shields.io/pypi/dm/unilvq.svg
   :target: https://img.shields.io/pypi/dm/unilvq.svg

.. image:: https://github.com/thieu1995/UniLVQ/actions/workflows/publish-package.yml/badge.svg
   :target: https://github.com/thieu1995/UniLVQ/actions/workflows/publish-package.yml

.. image:: https://pepy.tech/badge/unilvq
   :target: https://pepy.tech/project/unilvq

.. image:: https://readthedocs.org/projects/unilvq/badge/?version=latest
   :target: https://unilvq.readthedocs.io/en/latest/?badge=latest

.. image:: https://img.shields.io/badge/Chat-on%20Telegram-blue
   :target: https://t.me/+fRVCJGuGJg1mNDg1

.. image:: https://img.shields.io/badge/DOI-10.6084%2Fm9.figshare.29002685-blue
   :target: https://doi.org/10.6084/m9.figshare.29002685

.. image:: https://img.shields.io/badge/License-GPLv3-blue.svg
   :target: https://www.gnu.org/licenses/gpl-3.0


**UniLVQ** is an open-source Python library that provides a unified, extensible, and user-friendly
implementation of **Learning Vector Quantization (LVQ)** algorithms for supervised learning.
It supports both **classification** and **regression** tasks, and is designed to work seamlessly with the **scikit-learn API**.

Built on top of **NumPy** and **PyTorch**, UniLVQ combines rule-based and neural-inspired LVQ variants,
making it suitable for both research and practical applications.

* **Free software:** GNU General Public License (GPL) V3 license
* **Provided Estimators**:
    * **Classification**: `Lvq1Classifier`, `Lvq2Classifier`, `Lvq3Classifier`, `OptimizedLvq1Classifier`, `GlvqClassifier`, `GrlvqClassifier`, `LgmlvqClassifier`
    * **Regression**: `GlvqRegressor`, `GrlvqRegressor`
* **Supported performance metrics**: >= 67 (47 regressions and 20 classifications)
* **Documentation:** https://unilvq.readthedocs.io
* **Python versions:** >= 3.8.x
* **Dependencies:** numpy, scipy, scikit-learn, pandas, permetrics, torch


.. toctree::
   :maxdepth: 4
   :caption: Quick Start:

   pages/quick_start.rst

.. toctree::
   :maxdepth: 4
   :caption: Models API:

   pages/unilvq.rst

.. toctree::
   :maxdepth: 4
   :caption: Support:

   pages/support.rst



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
