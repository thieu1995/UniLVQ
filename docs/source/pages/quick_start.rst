============
Installation
============

* Install the `current PyPI release <https://pypi.python.org/pypi/unilvq />`_::

   $ pip install unilvq==0.2.0


* Install directly from source code::

   $ git clone https://github.com/thieu1995/UniLVQ.git
   $ cd UniLVQ
   $ python setup.py install

* In case, you want to install the development version from Github::

   $ pip install git+https://github.com/thieu1995/UniLVQ


After installation, you can import it as any other Python module::

   $ python
   >>> import unilvq
   >>> unilvq.__version__

========
Examples
========

In this section, we will explore the usage of the Adam-based GLVQ for multi-class classification problem::

    from sklearn.datasets import load_iris
    from unilvq import GlvqClassifier, Data


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
    model = GlvqClassifier(n_prototypes_per_class=2, epochs=1000, batch_size=16,
                           optim="Adam", optim_paras=None,
                           early_stopping=True, n_patience=10, epsilon=0.001, valid_rate=0.1,
                           seed=42, verbose=True, device=None)
    model.fit(data.X_train, data.y_train)

    ## Predict
    print("Predicted:", model.predict(data.X_test))

    ## Calculate some metrics
    print(model.score(X=data.X_test, y=data.y_test))
    print(model.scores(X=data.X_test, y=data.y_test, list_metrics=["PS", "RS", "NPV", "F1S", "F2S"]))
    print(model.evaluate(y_true=data.y_test, y_pred=model.predict(data.X_test), list_metrics=["F2S", "CKS", "FBS"]))


A real-world dataset contains features that vary in magnitudes, units, and range. We would suggest performing
normalization when the scale of a feature is irrelevant or misleading. Feature Scaling basically helps to normalize
the data within a particular range.

.. toctree::
   :maxdepth: 4

.. toctree::
   :maxdepth: 4

.. toctree::
   :maxdepth: 4
