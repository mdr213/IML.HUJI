from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    data_X = np.array_split(X, cv)
    data_y = np.array_split(y, cv)
    train_score_lst = []
    val_score_lst = []
    for i in range(0, cv):
        train_X = np.concatenate(np.delete(data_X, i, axis=0))
        train_y = np.concatenate(np.delete(data_y, i, axis=0))
        estimator.fit(train_X, train_y)
        train_pred = estimator.predict(train_X)
        test_pred = estimator.predict(data_X[i])
        train_score_lst.append(scoring(train_y, train_pred))
        val_score_lst.append(scoring(data_y[i], test_pred))
    return np.mean(train_score_lst), np.mean(val_score_lst)
