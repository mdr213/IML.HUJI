from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
pio.renderers.default = "browser"


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    X = np.linspace(-1.2, 2, n_samples)
    eps = np.random.normal(0, noise, n_samples)
    noiseless_y = (X + 3) * (X + 2) * (X + 1) * (X - 1) * (X - 2)
    y = noiseless_y + eps
    train_X, train_y, test_X, test_y = split_train_test(pd.DataFrame(X), pd.Series(y), 2 / 3)
    fig = go.Figure([go.Scatter(x=np.concatenate(np.asarray(train_X)), y=np.asarray(train_y), mode='markers',
                                    marker=dict(color='red'),
                                    name='Train set'),
                     go.Scatter(x=np.concatenate(np.asarray(test_X)), y=np.asarray(test_y), mode='markers',
                                    marker=dict(color='blue'),
                                    name='Test set'),
                     go.Scatter(x=X, y=noiseless_y, mode='lines+markers', marker=dict(color='black'), name='True model')])
    fig.update_layout(title=f'Question 1 - Dataset of model f(x) of size {n_samples} with noise {noise}')
    fig.show()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    train_avg_lst = []
    val_avg_lst = []
    x = np.linspace(0, 10, 11)
    for k in range(11):
        train_avg, val_avg = cross_validate(PolynomialFitting(k), np.asarray(train_X), np.asarray(train_y), mean_square_error)
        train_avg_lst.append(train_avg)
        val_avg_lst.append(val_avg)
    fig2 = go.Figure([go.Scatter(x=x, y=train_avg_lst, mode='lines+markers',
                                    marker=dict(color='red'),
                                    name='Average train error'),
                      go.Scatter(x=x, y=val_avg_lst, mode='lines+markers',
                                    marker=dict(color='blue'),
                                    name='Average validation error')])
    fig2.update_layout(title=f'Question 2 - Cross Validation with Polyfit of sizes 0-10 of size {n_samples} with noise {noise}')
    fig2.show()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    min_k = np.argmin(val_avg_lst, axis=0)
    poly = PolynomialFitting(min_k)
    poly.fit(np.concatenate(np.asarray(train_X)), np.asarray(train_y))
    y_pred = poly.predict(np.concatenate(np.asarray(test_X)))
    error = mean_square_error(np.asarray(test_y), y_pred)
    print(f'Question 3 - Polynomial degree with lowest validation error: {min_k}, with test error: {np.round(error, 2)}'
          f' of size {n_samples} with noise {noise}')


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    data_X, data_y = datasets.load_diabetes(return_X_y=True)
    train_X, train_y, test_X, test_y = split_train_test(pd.DataFrame(data_X), pd.Series(data_y), n_samples / data_y.size)

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    lambdas = 10 ** np.linspace(-4, 0.5, n_evaluations)
    ridge_scores = []
    lasso_scores = []
    ridge_vals = []
    lasso_vals = []
    for lam in lambdas:
        train_avg_ridge, val_avg_ridge = cross_validate(RidgeRegression(lam), np.asarray(train_X), np.asarray(train_y), mean_square_error)
        train_avg_lasso, val_avg_lasso = cross_validate(Lasso(lam, max_iter=10000), np.asarray(train_X), np.asarray(train_y), mean_square_error)
        ridge_scores.append(train_avg_ridge)
        ridge_vals.append(val_avg_ridge)
        lasso_scores.append(train_avg_lasso)
        lasso_vals.append(val_avg_lasso)

    fig = go.Figure([go.Scatter(x=lambdas,
                                y=ridge_scores, mode='markers',
                                marker=dict(color='red'),
                                name='Ridge train error'),
                     go.Scatter(x=lambdas,
                                y=ridge_vals, mode='markers',
                                marker=dict(color='orange'),
                                name='Ridge validation error'),
                     go.Scatter(x=lambdas,
                                y=lasso_scores, mode='markers',
                                marker=dict(color='purple'),
                                name='Lasso train error'),
                     go.Scatter(x=lambdas,
                                y=lasso_vals, mode='markers',
                                marker=dict(color='blue'),
                                name='Lasso validation error')])
    fig.update_layout(title='Question 7 - Train and validation error as a funcion of different regularization parameter')
    fig.show()

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    min_lam_ridge = lambdas[np.argmin(ridge_vals)]
    min_lam_lasso = lambdas[np.argmin(lasso_vals)]

    train_X = np.asarray(train_X)
    train_y = np.asarray(train_y)
    test_X = np.asarray(test_X)
    test_y = np.asarray(test_y)

    ridge = RidgeRegression(min_lam_ridge)
    ridge.fit(train_X, train_y)
    ridge_loss = ridge.loss(test_X, test_y)

    lasso = Lasso(min_lam_lasso)
    lasso.fit(train_X, train_y)
    y_pred_lasso = lasso.predict(test_X)
    lasso_loss = mean_square_error(test_y, y_pred_lasso)

    lin = LinearRegression()
    lin.fit(train_X, train_y)
    lin_loss = lin.loss(test_X, test_y)

    print('Question 8 -')
    print(f'Regularization parameter with lowest validation error for Ridge: {np.round(min_lam_ridge, 3)}, with error: {np.round(ridge_loss, 2)}')
    print(f'Regularization parameter with lowest validation error for Lasso: {np.round(min_lam_lasso, 3)}, with error: {np.round(lasso_loss, 2)}')
    print(f'Error for LS: {np.round(lin_loss, 2)}')


if __name__ == '__main__':
    np.random.seed(0)
    np.warnings.filterwarnings('ignore',
                               category=np.VisibleDeprecationWarning)
    select_polynomial_degree()
    select_polynomial_degree(noise=0)
    select_polynomial_degree(n_samples=1500, noise=10)
    select_regularization_parameter()
