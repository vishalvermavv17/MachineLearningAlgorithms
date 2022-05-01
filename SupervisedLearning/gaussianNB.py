import numpy as np
import pandas as pd


def update_class_label_count(y, classes_):
    classes_count_ = np.zeros((len(classes_), 1), dtype=int)
    for i in range(len(classes_)):
        classes_count_[i] = y[y == classes_[i]].shape[0]
    return classes_count_


def update_classes_prior_prob(classes_count_, sample_space_size):
    priors = classes_count_ / float(sample_space_size)
    return priors


class GaussianNB:
    """
    GaussianNB algorithm using gaussian probability density function.
        Parameters
        ----------
        priors : array-like of shape (n_classes,)
            Prior probabilities of the classes. If specified the priors are not adjusted according to the data.
        var_smoothing : float, default=1e-9
            Portion of the largest variance of all features that is added to variances for calculation stability.
        verbose : bool, default=False
            if True, print intermediate execution outputs for debug.

        Attributes
        -----------
        class_count_ : ndarray of shape (n_classes,)
            number of training samples observed in each class.
        class_prior_ : ndarray of shape (n_classes,)
            probability of each class.
        classes_ : ndarray of shape (n_classes,)
            class labels known to the classifier.
        epsilon_ : float
            absolute additive value to variances.
        n_features_in_ : int
            Number of features seen during fit.
        feature_names_in_ : ndarray of shape (n_features_in_,)
            Names of features seen during fit. Defined only when X has feature names that are all strings.
        var_ : ndarray of shape (n_classes, n_features)
            Variance of each feature per class.h'
        theta_ : ndarray of shape (n_classes, _features)
            mean of each feature per class.
    """

    def __init__(self, priors=None, var_smoothing=1e-9, verbose=False):
        self.priors = priors
        self.var_smoothing = var_smoothing
        self.verbose = verbose
        self.class_count_ = None
        self.class_prior_ = None
        self.classes_ = None
        self.epsilon_ = None
        self.n_features_in_ = None
        self.feature_names_in_ = None
        self.var_ = None
        self.theta_ = None

    def update_mean_variance(self, X, y):
        var_ = []
        theta_ = []

        for i in range(len(self.class_count_)):
            class_label_data = X[y == self.classes_[i]]
            var_.append(class_label_data.var(axis=0))
            theta_.append(class_label_data.mean(axis=0))

        self.var_ = var_
        self.theta_ = theta_

    def fit(self, X: pd.DataFrame, y):
        self.classes_ = np.unique(y)
        self.class_count_ = update_class_label_count(y, self.classes_)

        if self.verbose:
            print("No. of unique labels in dataset=[{}] and labels_name=[{}]".format(self.class_count_, self.classes_))

        if self.priors is None:
            self.priors = update_classes_prior_prob(self.class_count_, len(y))

        if self.verbose:
            print("Prior probabilities for class labels are: [{}]".format(self.priors))

        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = X.columns

        if self.verbose:
            print("No. of features in training dataset:[{}] and feature columns are: [{}]".format(
                self.n_features_in_, self.feature_names_in_))

        self.update_mean_variance(X, y)

    def _joint_log_likelihood(self, X):
        joint_log_likelihood = []
        for i in range(len(self.classes_)):
            class_i_log_prob = np.log(self.priors[i])
            x_ij_log_prob = -0.5 * np.sum(np.log(2 * np.pi * self.var_[i]))
            x_ij_log_prob -= 0.5 * np.sum(np.square((X - self.theta_[i])) / self.var_[i], axis=1)
            joint_log_likelihood.append(class_i_log_prob + x_ij_log_prob)

        print(joint_log_likelihood)
        joint_log_likelihood = np.array(joint_log_likelihood).T
        return joint_log_likelihood

    def predict(self, X):
        jll = self._joint_log_likelihood(X)
        return self.classes_[np.argmax(jll, axis=1)]
