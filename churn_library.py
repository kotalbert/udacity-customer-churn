"""
Module for Customer Churn ML Pipeline.
"""
from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def import_data(pth: str) -> pd.DataFrame:
    """
    Returns dataframe for the csv found at pth.

    :param pth: a path to the csv
    :return: pandas DataFrame with modeling data
    """

    return pd.read_csv(pth)


def perform_eda(df: pd.DataFrame) -> None:
    """
    Perform eda on df and save figures to `images` dorectory.

    :param df: pandas DataFrame with modeling data
    """
    _create_histogram(df, 'Churn', 'churn_hist.png')
    _create_histogram(df, 'Customer_Age', 'customer_age.png')
    _create_distplot(df, 'Total_Trans_Ct', 'total_trans_ct_distplot.png')
    _create_heatmap(df, 'correlation_heatmap.png')


def _create_histogram(df: pd.DataFrame, var_name: str, out_file_name: str) -> None:
    """
    Create histogram of a variable and save to png file.

    :param df: data frame for plot
    :param var_name: column name for plot
    :param out_file_name: name of output file
    """

    plt.figure(figsize=(20, 10))
    plt.title(var_name)
    df[var_name].hist()
    plt.savefig(f'./images/{out_file_name}')


def _create_distplot(df: pd.DataFrame, var_name: str, out_file_name: str) -> None:
    """
    Create distribution plot of a variable and save to png file.

    :param df: data frame for plot
    :param var_name: column name for plot
    :param out_file_name: name of output file
    """

    plt.figure(figsize=(20, 10))
    sns.distplot(df[var_name])
    plt.savefig(f'./images/{out_file_name}')


def _create_heatmap(df: pd.DataFrame, out_file_name: str) -> None:
    """
    Create variables correlation heatmap and save to png file.

    :param df: data frame for plot
    :param out_file_name: name of output file
    """

    plt.figure(figsize=(20, 10))
    sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig(f'./images/{out_file_name}')


def get_df_with_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a target variable 'Churn' and return new data frame with this variable added.

    :param df: df with data for model
    :return: pandas DataFrame with target variable added
    :raises ValueError: when data frame does not have required columns
    """

    try:
        assert 'Attrition_Flag' in df.columns
    except AssertionError:
        raise ValueError('Input data frame is missing required variables.')

    df_out = df.copy()
    df_out['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)
    return df_out


def encoder_helper(df: pd.DataFrame, category_lst: List[str], response: Optional[str] = None):
    """
    Helper function to turn each categorical column into a new column with proportion of churn for each category.

    :param df: input data
    :param category_lst: list of columns that contain categorical features
    :param response: (optional) string of response name to be used for
    :return: pandas dataframe with new columns
    """

    if response is None:
        response = 'Churn'

    df_out = df.copy()
    for var_name in category_lst:
        var_lst = []
        var_groups = df_out.groupby(var_name).mean()[response]
        for val in df_out[var_name]:
            var_lst.append(var_groups.loc[val])
        df_out[f'{var_name}_{response}'] = var_lst

    return df_out


def perform_feature_engineering(df, response):
    """
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    """


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    """
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    """
    pass


def feature_importance_plot(model, X_data, output_pth):
    """
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    """
    pass


def train_models(X_train, X_test, y_train, y_test):
    """
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    """
    pass
