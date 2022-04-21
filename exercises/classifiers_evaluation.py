from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi

pio.renderers.default = "browser"


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"), ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        X, Y = load_dataset("G:\My Drive\school\year two\semester B\iml\IML.HUJI\datasets\\" + f)

        def callback_func(fit: Perceptron, x: np.ndarray, y: int):
            losses.append(fit.loss(X, Y))

        # Fit Perceptron and record loss in each fit iteration
        losses = []
        per = Perceptron(include_intercept=True, callback=callback_func)
        per.fit(X, Y)

        # Plot figure of loss as function of fitting iteration
        fig = go.Figure((go.Scatter(x=np.linspace(1, 1000, 1000), y=np.asarray(losses), mode="lines",
                                    name="Mean Prediction",
                                    marker=dict(color="blue",)),))
        fig.update_layout(title_text=f'Part 1 - Loss as a function of fitting iterations - {n}')
        fig.show()


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black", showlegend=False)


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X, y = load_dataset("G:\My Drive\school\year two\semester B\iml\IML.HUJI\datasets\\" + f)

        # Fit models and predict over training set
        lin = LDA()
        gnb = GaussianNaiveBayes()
        lin.fit(X, y)
        y1 = lin.predict(X)
        gnb.fit(X, y)
        y2 = gnb.predict(X)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots
        from IMLearn.metrics import accuracy
        lin_acc = accuracy(y, y1)
        gnb_acc = accuracy(y, y2)
        symbols = np.array(["circle", "square", "triangle-up"])
        lims = np.array([X.min(axis=0), X.max(axis=0)]).T + np.array([-.4, .4])

        fig = make_subplots(rows=1, cols=2, subplot_titles=(f"Gaussian Naive Bayes with accuracy {gnb_acc}",
                                                            f"Linear Discriminant Analysis with accuracy {lin_acc}"))

        # Add traces for data-points setting symbols and colors
        fig.add_traces([decision_surface(gnb.predict, lims[0], lims[1], showscale=False),
                        decision_surface(lin.predict, lims[0], lims[1], showscale=False),
                         go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers',
                                    marker=dict(color=y,
                                                symbol=symbols[y]),
                                    showlegend=False),
                         go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers',
                                    marker=dict(color=y,
                                                symbol=symbols[y]),
                                    showlegend=False)],
                        rows=[1, 1, 1, 1], cols=[1, 2, 1, 2]) \
            .update_layout(
            title_text=f"Part 2 - Bayes Classifiers - {f.split('.')[0]}",)

        # Add `X` dots specifying fitted Gaussians' means
        fig.add_trace(go.Scatter(x=lin.mu_[:, 0], y=lin.mu_[:, 1], mode='markers',
                                    marker=dict(color="black", symbol="x"),
                                    showlegend=False), row=1, col=2,)
        fig.add_trace(go.Scatter(x=gnb.mu_[:, 0], y=gnb.mu_[:, 1], mode='markers',
                                 marker=dict(color="black",
                                             symbol="x"),
                                 showlegend=False), row=1, col=1,)

        # Add ellipses depicting the covariances of the fitted Gaussians
        fig.add_trace(get_ellipse(lin.mu_[0], lin.cov_), row=1, col=2,)
        fig.add_trace(get_ellipse(lin.mu_[1], lin.cov_), row=1, col=2, )
        fig.add_trace(get_ellipse(lin.mu_[2], lin.cov_), row=1, col=2, )
        fig.add_trace(get_ellipse(gnb.mu_[0], np.diag(gnb.vars_[0])), row=1, col=1, )
        fig.add_trace(get_ellipse(gnb.mu_[1], np.diag(gnb.vars_[1])), row=1, col=1, )
        fig.add_trace(get_ellipse(gnb.mu_[2], np.diag(gnb.vars_[2])), row=1, col=1, )
        fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
