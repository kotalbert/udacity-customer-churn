"""
Module for churn prediction library unit tests and logging utilities.
"""
import glob
import logging
import os
from typing import List

import pandas as pd

from churn_library import import_data, get_df_with_target, perform_eda, encoder_helper, perform_feature_engineering, \
    train_models

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

DATA_FILE_NAME = 'data/bank_data.csv'


def test_import():
    """
    Data import should create non-empty pandas DataFrame.
    """

    try:
        df = import_data(DATA_FILE_NAME)
    except FileNotFoundError as e:
        logging.error("Testing import_data: The file wasn't found")
        raise e

    try:
        assert isinstance(df, pd.DataFrame)
    except AssertionError as e:
        logging.error("Testing `import_data`: Data read does not produce correct data frame")
        raise e

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as e:
        logging.error("Testing `import_data`: The file doesn't appear to have rows and columns")
        raise e

    logging.info("Testing `import_data`: SUCCESS")


def test_import_data_integrity():
    """
    Imported data should contain expected columns.
    """

    df = import_data(DATA_FILE_NAME)

    columns_expected = {
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category',
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio'
    }

    columns_actual = set(df.columns)
    try:
        assert columns_expected.issubset(columns_actual)
    except AssertionError as e:
        logging.error("Testing `import_data` integrity: Data file is missing required columns")
        raise e

    logging.info("Testing `import_data` integrity: SUCCESS")


def test_get_df_with_target():
    """
    Function should return data frame with target variable 'Churn' added.
    """

    df = import_data(DATA_FILE_NAME)
    df_with_target = get_df_with_target(df)

    expected_columns_list = list(df.columns)
    expected_columns_list.append('Churn')

    expected_columns_set = set(expected_columns_list)
    actual_columns_set = set(df_with_target.columns)

    try:
        assert expected_columns_set == actual_columns_set
    except AssertionError as e:
        logging.error("Testing `get_df_with_target`: target variable is missing from output df")
        raise e

    logging.info("Testing `get_df_with_target`SUCCESS")


def _clean_dir(pth: str, ptrn: str = '*.png') -> None:
    """
    Utility to clear directory before testing if files are presetn.

    :param pth: path to dir to be cleared
    :param ptrn: optional, fille pattern to be cleared, defaults to '*.png'
    """

    for img in glob.glob(os.path.join(pth, ptrn)):
        os.remove(img)


def _check_expected_files(pth: str, files_expected: List[str]) -> None:
    """
    Helper to check if expected files are in directory.

    :param pth:
    :param files_expected:
    :raises AssertionError: if not all files present
    """

    files_actual = set(os.listdir(pth))
    assert files_actual.issuperset(set(files_expected))


def test_eda():
    """
    Test if `perform_eda` function is generating expected plots.
    """

    _clean_dir('images')
    df = get_df_with_target(import_data(DATA_FILE_NAME))
    perform_eda(df)

    plots_expected = ['churn_hist.png', 'customer_age.png', 'total_trans_ct_distplot.png', 'correlation_heatmap.png']

    try:
        _check_expected_files('images', plots_expected)
        logging.info("Testing `perform_eda`: SUCCESS")
    except AssertionError as e:
        logging.error("Testing `perform_eda`: expected plot was not generated")
        raise e


def test_encoder_helper():
    """Function should add new categorical columns to data frame."""

    df = get_df_with_target(import_data(DATA_FILE_NAME))

    category_lst = ['Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category']
    df_with_cat_vars = encoder_helper(df, category_lst)

    expected_category_variables = [f'{var_name}_Churn' for var_name in category_lst]

    try:
        assert set(df_with_cat_vars).issuperset(expected_category_variables)
    except AssertionError as e:
        logging.error("Testing `encoder_helper`: expected variable not added to df")
        raise e
    logging.info("Testing `encoder_helper`: SUCCESS")


def test_perform_feature_engineering():
    """
    Feature engineering should create test/train split of the input data.
    """

    category_lst = ['Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category']
    df = encoder_helper(get_df_with_target(import_data(DATA_FILE_NAME)), category_lst)
    x_train, x_test, y_train, y_test = perform_feature_engineering(df)

    try:
        assert isinstance(x_train, pd.DataFrame)
        assert isinstance(x_test, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        assert isinstance(y_test, pd.Series)
    except AssertionError as e:
        logging.error('Testing `perform_feature_engineering: expected data not created.')
        raise e
    logging.info('Testing `perform_feature_engineering`: SUCCESS')


def test_train_models():
    """
    Artifacts from model training should be created.
    """

    _clean_dir('images')
    _clean_dir('models', '*.pkl')

    category_lst = ['Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category']
    df = encoder_helper(get_df_with_target(import_data(DATA_FILE_NAME)), category_lst)
    x_train, x_test, y_train, y_test = perform_feature_engineering(df)

    train_models(x_train, x_test, y_train, y_test)

    plots_expected = ['feature_importance_plot.png', 'logistic_regresion_classification_report.png',
                      'random_forest_classification_report.png']
    try:
        _check_expected_files('images', plots_expected)
    except AssertionError as e:
        logging.error("Testing `train_models`: expected plots were not generated")
        raise e

    models_expected = ['logistic_model.pkl', 'rfc_model.pkl']

    try:
        _check_expected_files('models', models_expected)
    except AssertionError as e:
        logging.error("Testing `train_models`: expected model binaries were not generated")
        raise e

    logging.info('Testing `train_models`: SUCCESS')


if __name__ == "__main__":
    test_import()
    test_import_data_integrity()
    test_get_df_with_target()
    test_eda()
    test_encoder_helper()
    test_train_models()
