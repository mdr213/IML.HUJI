from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
from IMLearn.metrics import loss_functions
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"
pio.renderers.default = "browser"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    df = pd.read_csv(filename)
    zip_dict = {}

    def add_to_dict(x):
        if x['zipcode'] not in zip_dict:
            zip_dict[x['zipcode']] = [x['price']]
        else:
            zip_dict[x['zipcode']].append(x['price'])

    df.apply(add_to_dict, axis=1)
    for key in zip_dict:
        zip_dict[key] = sum(zip_dict[key]) // len(zip_dict[key])
    zips = df['zipcode'].copy()
    for i in range(len(zips)):
        if zips[i] > 0:
            zips[i] = zip_dict[zips[i]]
    df.insert(0, 'zipcodes', zips)
    df = df.drop(columns=['id', 'date', 'lat', 'long', 'sqft_living15', 'sqft_lot15', 'zipcode'])
    df['yr_renovated'] = df.apply(
        lambda x: x['yr_built'] if x['yr_renovated'] == 0 else x['yr_renovated'], axis=1)
    df = df.dropna()
    prices = df['price'].copy()
    df = df.drop(columns='price')
    return df, prices


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    y_std = y.std()
    corr_lst = []
    for column in X:
        df = X[[column]].copy()
        df.insert(1, 'y', y)
        c_std = X[column].std()
        corr = (df.cov() / (y_std * c_std))[column][1]
        corr_lst.append((column, corr))
        fig = go.Figure(layout=dict(title=f'The correlation between {column} and prices is {corr}'))
        fig.add_trace(go.Scatter(x=X[column], y=y, name=column, mode="markers"))
        fig.update_yaxes(title_text='House Prices')
        fig.update_xaxes(title_text=column)
        fig.write_image(output_path + f'\\{column}.jpg')
    # print(corr_lst)


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    X, y = load_data("G:\My Drive\school\year two\semester B\iml\IML.HUJI\datasets\house_prices.csv")

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(X, y, 'G:\My Drive\school\year two\semester B\iml\exercises\ex2')

    # Question 3 - Split samples into training- and testing sets.
    x_train, y_train, x_test, y_test = split_train_test(X, y)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    linreg = LinearRegression()
    loss_lst = []
    std_lst = []
    for p in range(10, 101):
        curr_loss = []
        for i in range(10):
            sample_x = x_train.sample(frac=p/100, random_state=i)
            sample_y = y_train.sample(frac=p/100, random_state=i)
            linreg.fit(sample_x.to_numpy(), sample_y.to_numpy())
            curr_loss.append(linreg.loss(x_test.to_numpy(), y_test.to_numpy()))
        mean = np.mean(curr_loss)
        std = np.std(curr_loss)
        loss_lst.append(mean)
        std_lst.append(std)
    mean_pred, var_pred = np.asarray(loss_lst), np.asarray(std_lst)
    x = np.linspace(10, 101, 92)
    fig = go.Figure((go.Scatter(x=x, y=mean_pred, mode="markers+lines",
                                    name="Mean Prediction",
                                    line=dict(dash="dash"),
                                    marker=dict(color="green", opacity=.7)),
                         go.Scatter(x=x, y=mean_pred - 2 * var_pred,
                                    fill=None, mode="lines",
                                    line=dict(color="lightgrey"),
                                    showlegend=False),
                         go.Scatter(x=x, y=mean_pred + 2 * var_pred,
                                    fill='tonexty', mode="lines",
                                    line=dict(color="lightgrey"),
                                    showlegend=False),))
    fig.update_layout(title_text='Question 4 - MSE as a function of p')
    fig.show()


    # Quiz Q2
    # y_true = np.array([279000, 432000, 326000, 333000, 437400, 555950])
    # y_pred = np.array(
    #     [199000.37562541, 452589.25533196, 345267.48129011, 345856.57131275,
    #      563867.1347574, 395102.94362135])
    # print(loss_functions.mean_square_error(y_true, y_pred))
