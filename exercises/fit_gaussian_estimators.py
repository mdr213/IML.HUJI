from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import sys
sys.path.append("../")
from scipy.stats import norm
from utils import *
pio.renderers.default = "browser"
pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    X = np.random.normal(10, 1, 1000)
    estimator = UnivariateGaussian()
    estimator.fit(X)
    print('(expectation, variance): ', (estimator.mu_, estimator.var_))

    # Question 2 - Empirically showing sample mean is consistent
    x_axis = np.linspace(10, 1000, 100)
    y_axis = []
    estimator_sample = UnivariateGaussian()
    for i in x_axis:
        sample = X[0:int(i)]
        estimator_sample.fit(sample)
        y_axis.append(np.abs(10 - estimator_sample.mu_))

    fig1 = make_subplots(rows=1, cols=1) \
        .add_traces([go.Scatter(x=x_axis, y=y_axis, mode='lines',
                                marker=dict(color="black"),
                                showlegend=False)]) \
        .update_layout(
        title_text=r"$\text{Question 2 - Correlation between loss and sample sizes}$",
        xaxis={"title": "x - Sample Sizes"},
        yaxis={"title": "y - Loss"})
    fig1.show()

    # Question 3 - Plotting Empirical PDF of fitted model
    y_axis2 = estimator.pdf(X)

    fig2 = make_subplots(rows=1, cols=1) \
        .add_traces([go.Scatter(x=X, y=y_axis2, mode='markers',
                                marker=dict(color="black"),
                                showlegend=False)]) \
        .update_layout(
        title_text=r"$\text{Question 3 - Correlation between loss and sample sizes}$",
        xaxis={"title": "x - Sample Values"},
        yaxis={"title": "y - PDF Values "})
    fig2.show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mu = np.array([0, 0, 4, 0])
    sigma = np.array([[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]])
    X = np.random.multivariate_normal(mu, sigma, 1000)
    estimator = MultivariateGaussian()
    estimator.fit(X)
    print('estimated mean: ', estimator.mu_)
    print('estimated covariance: ', estimator.cov_)

    # Question 5 - Likelihood evaluation
    f1 = np.linspace(-10, 10, 200)
    f3 = np.linspace(-10, 10, 200)
    z = []
    max_like = -100000
    max_i = 0
    max_j = 0
    for i in f1:
        temp = []
        for j in f3:
            mu = np.array([i, 0, j, 0])
            log_l = MultivariateGaussian.log_likelihood(mu, sigma, X)
            temp.append(log_l)
            if log_l > max_like:
                max_like = log_l
                max_i = i
                max_j = j
        z.append(temp)

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=f3,
        y=f1,
        colorscale='Viridis'))
    fig.update_layout(
        title='Question 5 - Log Likelihood Heatmap',
        yaxis_title='y - f1 values',
        xaxis_title='x - f3 values')
    fig.show()

    # Question 6 - Maximum likelihood
    # print(np.amax(z))
    print('(f1 value, f3 value): ', (max_i.round(4), max_j.round(4)))


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
    # n = np.array(
    #     [1, 5, 2, 3, 8, -4, -2, 5, 1, 10, -10, 4, 5, 2, 7, 1, 1, 3, 2, -1,
    #      -3, 1, -4, 1, 2, 1,
    #      -4, -4, 1, 3, 2, 6, -6, 8, 3, -6, 4, 1, -2, 3, 1, 4, 1, 4, -2, 3,
    #      -1, 0, 3, 5, 0, -2])
    # s = UnivariateGaussian()
    # s.fit(n)
    # print(s.log_likelihood(10, 1, n))
