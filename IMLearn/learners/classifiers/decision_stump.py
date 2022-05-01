from __future__ import annotations
from typing import Tuple, NoReturn
from ...base import BaseEstimator
import numpy as np
from itertools import product
from ...metrics import misclassification_error


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """
    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        m, d = X.shape
        loss_dict = dict()
        for feat in range(d):
            plus_thr, plus_err = self._find_threshold(X[:, feat], y, 1)
            minus_thr, minus_err = self._find_threshold(X[:, feat], y, -1)
            if plus_err > minus_err:
                loss_dict[plus_err] = [feat, plus_thr, 1]
            else:
                loss_dict[minus_err] = [feat, minus_thr, -1]
        minn = min(loss_dict)
        self.j_ = loss_dict[minn][0]
        self.threshold_ = loss_dict[minn][1]
        self.sign_ = loss_dict[minn][2]

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        # y_pred = np.zeros(X.size)
        y_pred = np.asarray([self.sign_ if x >= self.threshold_ else -self.sign_ for x in X])
        # for i in range(X.size):
        #     y_pred[i] = self.sign_ if X[i][self.j_] >= self.threshold_ else -self.sign_
        return y_pred

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """
        thr_err = 1.0
        thr = 0.0
        y_temp = np.zeros(values.size)
        for i in range(values.size):
            for j in range(values.size):
                y_temp[j] = sign if values[j] >= values[i] else -sign
            temp_err = misclassification_error(labels, y_temp, True)
            if temp_err < thr_err:
                thr = values[i]
                thr_err = temp_err
        return thr, thr_err

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        y_pred = self.predict(X)
        return misclassification_error(y, y_pred)
