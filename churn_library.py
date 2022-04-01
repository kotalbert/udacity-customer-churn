"""
Module for Customer Churn ML Pipeline.
"""
from typing import List, Optional, Tuple, Union, Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV

# type aliases
DF = pd.DataFrame
SR = pd.Series
RFC = RandomForestClassifier
LRC = LogisticRegression


def import_data(pth: str) -> DF:
    """
    Returns dataframe for the csv found at pth.

    :param pth: a path to the csv
    :return: pandas DataFrame with modeling data
    """

    return pd.read_csv(pth)


def perform_eda(data_frame: DF) -> None:
    """
    Perform eda on df and save figures to `images` dorectory.

    :param data_frame: pandas DataFrame with modeling data
    """
    _create_histogram(data_frame, 'Churn', 'churn_hist.png')
    _create_histogram(data_frame, 'Customer_Age', 'customer_age.png')
    _create_distplot(data_frame, 'Total_Trans_Ct', 'total_trans_ct_distplot.png')
    _create_heatmap(data_frame, 'correlation_heatmap.png')


def _create_histogram(df: DF, var_name: str, out_file_name: str) -> None:
    """
    Create histogram of a variable and save to png file.

    :param df: data frame for plot
    :param var_name: column name for plot
    :param out_file_name: name of output file
    """

    plt.title(var_name)
    df[var_name].hist()
    plt.savefig(f'./images/{out_file_name}')


def _create_distplot(df: DF, var_name: str, out_file_name: str) -> None:
    """
    Create distribution plot of a variable and save to png file.

    :param df: data frame for plot
    :param var_name: column name for plot
    :param out_file_name: name of output file
    """

    plt.figure(figsize=(20, 10))
    sns.distplot(df[var_name])
    plt.savefig(f'./images/{out_file_name}')


def _create_heatmap(df: DF, out_file_name: str) -> None:
    """
    Create variables correlation heatmap and save to png file.

    :param df: data frame for plot
    :param out_file_name: name of output file
    """

    plt.figure(figsize=(20, 10))
    sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig(f'./images/{out_file_name}')


def get_df_with_target(df: DF) -> DF:
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


def encoder_helper(df: DF, category_lst: List[str], response: Optional[str] = 'Churn'):
    """
    Helper function to turn each categorical column into a new column with proportion of churn for each category.

    :param df: input data
    :param category_lst: list of columns that contain categorical features
    :param response: (optional, defaults to 'Churn') string of response name to be used for crating columns
    :return: pandas dataframe with new columns
    """

    df_out = df.copy()
    for var_name in category_lst:
        var_lst = []
        var_groups = df_out.groupby(var_name).mean()[response]
        for val in df_out[var_name]:
            var_lst.append(var_groups.loc[val])
        df_out[f'{var_name}_{response}'] = var_lst

    return df_out


def perform_feature_engineering(df: DF, response: Optional[str] = 'Churn', keep_cols: Optional[List] = None) \
        -> Tuple[DF, DF, SR, SR]:
    """
    Keep relevant variables and split data to train and test samples.

    :param df: input data
    :param response: (optional, defaults to 'Churn') string of response name to be used for crating columns
    :param keep_cols: (optional, default list of columns to keep) list of column names
    :return: tuple of train/test data frames and train/test responses:
    output:
         X training data
         X testing data
         y training data
         y testing data
    """

    if keep_cols is None:
        keep_cols = ['Customer_Age', 'Dependent_count', 'Months_on_book',
                     'Total_Relationship_Count', 'Months_Inactive_12_mon',
                     'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
                     'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
                     'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
                     'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn',
                     'Income_Category_Churn', 'Card_Category_Churn']

        x = df[keep_cols]
        y = df[response]

        return train_test_split(x, y, test_size=0.3, random_state=42)


def classification_report_image(y_train, y_test, y_train_preds_lr, y_train_preds_rf, y_test_preds_lr,
                                y_test_preds_rf) -> None:
    """
    Produces classification report for training and testing results and stores report as image in images folder.

    :param y_train: training response values
    :param y_test:  test response values
    :param y_train_preds_lr: training predictions from logistic regression
    :param y_train_preds_rf: training predictions from random forest
    :param y_test_preds_lr: test predictions from logistic regression
    :param y_test_preds_rf: test predictions from random forest
    """

    _create_classification_report('Logistic Regression', y_train, y_train_preds_lr, y_test, y_test_preds_lr)
    _create_classification_report('Random Forest', y_train, y_train_preds_rf, y_test, y_test_preds_rf)


def _slugify(s: str) -> str:
    """
    Helper to create consistent file name from a string.
    """

    return s.lower().replace(' ', '_')


def _create_classification_report(model_name: str, y_train, y_train_pred, y_test, y_test_pred) -> None:
    """
    Create a classification report and write as image.

    :param model_name:
    :param y_train:
    :param y_train_pred:
    :param y_test:
    :param y_test_pred:
    """

    plt.text(0.01, 1.25, str(f'{model_name} Train'), {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_pred)), {'fontsize': 10},
             fontproperties='monospace')
    plt.text(0.01, 0.6, str(f'{model_name} Test'), {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_pred)), {'fontsize': 10},
             fontproperties='monospace')
    plt.axis('off')
    output_file_name = _slugify(model_name)
    plt.savefig(f'images/{output_file_name}_classification_report.png', )


def feature_importance_plot(model: Any, x_data: DF, output_pth: str) -> None:
    """
    Creates and stores the feature importances in pth.

    :param model: model object containing feature_importances_
    :param x_data: pandas dataframe of X values
    :param output_pth: path to store the figure
    """

    # object must have feature_importances_ attribute
    assert hasattr(model, 'feature_importances_')

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    names = [x_data.columns[i] for i in indices]

    plt.title("Feature Importance")
    plt.ylabel('Importance')
    plt.bar(range(x_data.shape[1]), importances[indices])
    plt.xticks(range(x_data.shape[1]), names, rotation=90)
    plt.savefig(f'{output_pth}')


def train_models(x_train: DF, x_test: DF, y_train: SR, y_test: SR) -> None:
    """
    Train, store model results: images + scores, and store serialized models.

    :param x_train: X training data
    :param x_test: X testing data
    :param y_train: y training data
    :param y_test: y testing data
    """

    rfc = _train_rfc(x_train, y_train)
    lrc = _train_lrc(x_train, y_train)

    _dump_model(rfc, 'models/rfc_model.pkl')
    _dump_model(lrc, 'models/logistic_model.pkl')

    y_train_preds_lr = lrc.predict(x_train)
    y_train_preds_rf = rfc.predict(x_train)
    y_test_preds_lr = lrc.predict(x_test)
    y_test_preds_rf = rfc.predict(x_test)

    classification_report_image(y_train, y_test, y_train_preds_lr, y_train_preds_rf, y_test_preds_lr, y_test_preds_rf)
    x_data = x_train.append(x_test)
    feature_importance_plot(rfc, x_data, 'images/feature_importance_plot.png')


def _dump_model(cls: Union[RFC, LRC], output_filename) -> None:
    """
    Serialize model and dump to file.

    :param cls: model to serialize
    :param output_filename:
    """

    joblib.dump(cls, output_filename)


def _train_rfc(x_train: DF, y_train: SR) -> RFC:
    """
    Train Random Forest Classifier.
    """

    rfc = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(x_train, y_train)
    return cv_rfc.best_estimator_


def _train_lrc(x_train: DF, y_train: SR) -> LRC:
    """
    Train Logistic Regression Classifier.
    """

    lrc = LogisticRegression()
    return lrc.fit(x_train, y_train)
