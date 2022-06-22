import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"
pio.renderers.default = "browser"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename, parse_dates=['Date']).dropna().drop_duplicates()
    df['DayOfYear'] = df['Date'].dt.dayofyear
    df.drop(df[df.Temp < -20].index, inplace=True)
    # print(df)
    return df


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    df = load_data("G:\My Drive\school\year two\semester B\iml\IML.HUJI\datasets\City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    df["Year"] = df["Year"].astype(str)
    df_israel = df[df.Country == 'Israel']
    fig = px.scatter(df_israel, x='DayOfYear', y='Temp', color='Year')
    fig.update_layout(title_text='Question 2 - Temperature as a function of day of year concluded of the year 1995-2007')
    fig.show()

    df_israel_m = df_israel.groupby('Month').agg('std')
    fig2 = px.bar(df_israel_m, y='Temp')
    fig2.update_layout(title_text='Question 2 - Temperature standard deviation as a function of the months')
    fig2.show()

    # Question 3 - Exploring differences between countries
    df_group = df.groupby(['Country', 'Month']).Temp.agg(['mean', 'std']).reset_index()
    fig3 = px.line(df_group, x=['Month'], y='mean', error_y='std', color='Country')
    fig3.update_layout(title_text='Question 3 - The mean and standard deviation of temperatures in every country according to the months')
    fig3.show()

    # Question 4 - Fitting model for different values of `k`
    train_x, train_y, test_x, test_y = split_train_test(df_israel['DayOfYear'], df_israel['Temp'])
    loss_lst = []
    for k in range(1, 11):
        polyfit = PolynomialFitting(k)
        polyfit.fit(train_x.to_numpy(), train_y.to_numpy())
        loss_lst.append(round(polyfit.loss(test_x.to_numpy(), test_y.to_numpy()), 2))
    fig4 = px.bar(x=np.linspace(1, 10, 10), y=loss_lst)
    fig4.update_layout(title_text='Question 4 - The loss as a function of different k values')
    fig4.update_yaxes(title_text='Loss')
    fig4.update_xaxes(title_text='K values')
    fig4.show()
    print(loss_lst)


    # Question 5 - Evaluating fitted model on different countries
    polyfit5 = PolynomialFitting(5)
    polyfit5.fit(df_israel['DayOfYear'], df_israel['Temp'])
    loss_lst_c = []
    countries = df.Country.unique().tolist()
    countries.remove('Israel')
    for country in countries:
        country_df = df[df.Country == country]
        loss_lst_c.append(polyfit5.loss(country_df['DayOfYear'], country_df['Temp']))
    fig5 = px.bar(x=countries, y=loss_lst_c)
    fig5.update_layout(title_text='Question 5 - The loss of other countries according to a model based on Israel')
    fig5.update_yaxes(title_text='Loss')
    fig5.update_xaxes(title_text='Country')
    fig5.show()
