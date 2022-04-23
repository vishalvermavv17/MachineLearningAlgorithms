import pandas as pd
import numpy as np


class LogisticRegressionUsingGD:
    """Logistic Regression Using Gradient Descent.

            Parameters
            ----------
            alpha : float
                Learning Rate
            n_iterations : int
                No of passes over the training set.

            Attributes
            __________
            w_ : weights after the model.
            cost_ : total error of the model after each iteration.

        """

    def __init__(self, alpha=0.05, n_iterations=100):
        self.alpha = alpha
        self.n_iterations = n_iterations

    def sigmoid(self, z):
        sigmoid_z = 1 / (1 + np.exp(-z))
        return sigmoid_z

    def fit(self, x, y):
        """Fit the training data.

                Parameters
                ----------
                x : array-like, shape = [n_samples, n_features]
                    Training samples
                y : array-like, shape = [n_samples, n_target_values]
                    Target values

                Returns
                -------
                self : object
                """
        # Add column with constant value = 1 to represent intercept in linear regression equation
        x = np.c_[np.ones(x.shape[0]), x]
        y = np.c_[y]

        self.cost_ = []
        self.w_ = np.zeros((x.shape[1], 1))
        m = x.shape[0]

        for _ in range(self.n_iterations):
            y_pred = self.sigmoid(np.dot(x, self.w_))
            self.w_ -= self.alpha * (1/m) * np.dot(x.T, y_pred - y)
            cost = (-1/2*m) * (np.dot(y.T, np.log(y_pred)) + np.dot((1-y).T, np.log(1-y_pred)))
            self.cost_.append(cost)
        return self

    def predict(self, x):
        """ Predicts the value after the model has been trained.
        Parameters
        ----------
        x : array-like, shape = [n_samples, n_features]
            Test samples
        Returns
        -------
        Predicted value
        """
        # Add column with constant value = 1 to represent intercept in linear regression equation
        x = np.c_[np.ones(x.shape[0]), x]
        z = self.sigmoid(np.dot(x, self.w_))
        predictions = []

        for i in self.sigmoid(z):
            if i < 0.5:
                predictions.append(i)
            else:
                predictions.append(i)
        return predictions
