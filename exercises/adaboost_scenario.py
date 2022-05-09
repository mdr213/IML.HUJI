import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots

pio.renderers.default = "browser"


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    ada = AdaBoost(DecisionStump, n_learners)
    ada.fit(train_X, train_y)
    loss_lst_test = []
    loss_lst_train = []
    x_axis = list(range(n_learners))
    for i in range(1, n_learners + 1):
        loss_lst_test.append(ada.partial_loss(test_X, test_y, i))
        loss_lst_train.append(ada.partial_loss(train_X, train_y, i))

    fig = go.Figure([go.Scatter(x=x_axis, y=loss_lst_train, mode='lines', name='Train Loss'),
                     go.Scatter(x=x_axis, y=loss_lst_test, mode='lines', name='Test Loss')])
    fig.update_layout(title=f'Question 1 - Loss as a function of number of learners with noise {noise}', xaxis_title='Number of Learners', yaxis_title='Loss values')
    fig.show()

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])

    lst = []
    fig2 = make_subplots(rows=2, cols=2, subplot_titles=(
        [f'Ensemble with {t} learners' for t in T]))
    for t in T:
        ada.iterations_ = t
        lst.append(decision_surface(ada.predict, lims[0], lims[1],
                                    showscale=False))

    fig2.add_traces(lst +
                    [go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode='markers',
                                marker=dict(color=test_y), showlegend=False),
                     go.Scatter(x=test_X[:, 0], y=test_X[:, 1],
                                mode='markers',
                                marker=dict(color=test_y),
                                            showlegend=False),
                     go.Scatter(x=test_X[:, 0], y=test_X[:, 1],
                                mode='markers',
                                marker=dict(color=test_y),
                                            showlegend=False),
                     go.Scatter(x=test_X[:, 0], y=test_X[:, 1],
                                mode='markers',
                                marker=dict(color=test_y),
                                            showlegend=False)],
                    rows=[1, 1, 2, 2, 1, 1, 2, 2], cols=[1, 2, 1, 2, 1, 2, 1, 2]) \
        .update_layout(height=600, width=1200,
        title_text=f'Question 2 - Boosting with different ensemble sizes with noise {noise}')
    fig2.update_layout(yaxis1=dict(range=[-1, 1]), yaxis2=dict(range=[-1, 1]), yaxis3=dict(range=[-1, 1]),
                       yaxis4=dict(range=[-1, 1]), xaxis1=dict(range=[-1, 1]), xaxis2=dict(range=[-1, 1]),
                       xaxis3=dict(range=[-1, 1]), xaxis4=dict(range=[-1, 1]))
    fig2.show()

    # Question 3: Decision surface of best performing ensemble
    min_err = np.argmin(loss_lst_test)
    ada.iterations_ = min_err
    fig3 = go.Figure(
        [decision_surface(ada.predict, lims[0], lims[1], showscale=False),
         go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode='markers',
                    marker=dict(color=test_y, colorscale=[custom[0], custom[-1]]),
                    showlegend=False)],
        layout=go.Layout(
            title=f'Question 3 - minimal loss of ensemble of size {min_err + 1} with accuracy of {1 - loss_lst_test[min_err]} with noise {noise}'))
    fig3.update_layout(xaxis_range=[-1, 1], yaxis_range=[-1, 1])
    fig3.show()

    # Question 4: Decision surface with weighted samples
    ada.iterations_ = n_learners
    fig4 = go.Figure(
        [decision_surface(ada.predict, lims[0], lims[1], showscale=False),
         go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode='markers',
                    marker=dict(color=train_y, colorscale=[custom[0], custom[-1]], size=ada.D_ / np.max(ada.D_) * 5),
                    showlegend=False)],
        layout=go.Layout(
            title=f'Question 4 - the data with bubble sizes according to weights with noise {noise}'))
    fig4.update_layout(xaxis_range=[-1, 1], yaxis_range=[-1, 1])
    fig4.show()


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0)
    fit_and_evaluate_adaboost(0.4)
