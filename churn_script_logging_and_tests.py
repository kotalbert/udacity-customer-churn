import glob
import logging
import os

import pandas as pd

from churn_library import import_data, get_df_with_target, perform_eda

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

DATA_FILE_NAME = "./data/bank_data.csv"


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
        logging.error('Testing `import_data`: Data read does not produce correct data frame')
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
        logging.error('Testing `import_data` integrity: Data file is missing required columns')
        raise e

    logging.info('Testing `import_data` integrity: SUCCESS')


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
        logging.error('Testing `get_df_with_target`: target variable is missing from output df')
        raise e

    logging.info('Testing `get_df_with_target`SUCCESS')


def _clean_dir(pth: str) -> None:
    """
    Utility to clear directory before testing if files are presetn.

    :param pth: path to dir
    """

    for img in glob.glob(os.path.join(pth, '*.png')):
        os.remove(img)


def test_eda():
    """
    Test if `perform_eda` function is generating expected plots.
    """

    _clean_dir('images')
    df = get_df_with_target(import_data(DATA_FILE_NAME))
    perform_eda(df)

    plots_expected = ['churn_hist.png', 'customer_age.png', 'total_trans_ct_distplot.png', 'correlation_heatmap.png']

    try:
        for ei in plots_expected:
            assert os.path.isfile(os.path.join('images', ei))
        logging.info(f'Testing perform_eda: SUCCESS')

    except AssertionError as e:
        logging.error(f'Testing `perform_eda`: expected plot was not generated')
        raise e


#
#
# def test_encoder_helper(encoder_helper):
#     """
#     test encoder helper
#     """
#     pass
#
#
# def test_perform_feature_engineering(perform_feature_engineering):
#     """
#     test perform_feature_engineering
#     """
#     pass
#
#
# def test_train_models(train_models):
#     """
#     test train_models
#     """
#     pass


if __name__ == "__main__":
    test_import()
    test_import_data_integrity()
    test_get_df_with_target()
    test_eda()
