import pandas as pd
import numpy as np
from dateutil.parser import parse
from print_logger import print


def is_date(value):
    """
        Checks whether a value is a date by attempting to parse it.
    """
    try:
        parse(str(value))
        return True
    except:
        return False


def is_nat(value):
    """
        Checks whether a value is an NaT (not a time).
    """
    try:
        if np.isnat(np.datetime64(str(value))):
            return True
        else:
            return False
    except:
        return False


def is_date_col(df_col: pd.Series):
    """
        Checks whether random 1000 values in a column convert to dates.
        If yes, returns that this is a date column.
    """
    date_col = False
    # Retrieve a random sample of 1000 values to check whether dates
    sample_size = 1000
    if df_col.shape[0] < 1000:
        sample_size = df_col.shape[0]

    top_vals = df_col.sample(sample_size).values
    for val in top_vals:
        if is_date(val) or is_nat(val):
            date_col = True
            break

    return date_col


def is_category_col(df_col: pd.Series):
    """
        Checks whether an object (string) column should be converted to a category.
        If on average there are two values for every category, column should be
        a category.
    """
    num_unique_values = len(df_col.unique())
    num_total_values = len(df_col)
    # Check we will get at least half the values removed by changing to category
    return bool((num_unique_values / num_total_values) < 0.5)


def is_int8_col(df_col: pd.Series):
    """ Checks a numeric column only contains 0s and 1s """
    return df_col.isin([0, 1]).all()


def optimise_df(df: pd.Series, verbose=False):
    """
        Works through the columns of a dataframe and optimises for memory where
        possible.
    """

    optimised_df = pd.DataFrame()

    for col in df.columns.values:
        print('Field:', col)
        existing_col_type = str(df[col].dtype)
        if verbose:
            print('type:', existing_col_type)

        if col == 'symbol':
            calculated_col_type = 'category'
        else:
            calculated_col_type = get_col_type(df[col])

        if calculated_col_type == 'date':
            if verbose:
                print('Coverting to date')
            optimised_df[col] = pd.to_datetime(df[col], errors='coerce')
        elif calculated_col_type == 'category' and existing_col_type != 'category':
            # fill missing values with NA and convert
            if verbose:
                print('Filling missing values with NA and converting to category')
            optimised_df[col] = df[col].fillna('NA').astype('category')
        elif calculated_col_type == 'category' and existing_col_type == 'category':
            # Check whether this category has missing values
            if verbose:
                print('Column is already category - checking for missing values with NA')
            if df[col].isna().sum() > 0:
                # Missing values found - add NA to category values
                df[col].cat.add_categories('NA', inplace=True)
                # Fill missing with NA
                df[col].fillna('NA', inplace=True)

            optimised_df[col] = df[col]
        elif calculated_col_type == 'int8':
            if verbose:
                print('Coverting to int8')
            optimised_df[col] = df[col].astype('int8', errors='ignore')
        elif calculated_col_type == 'int32':
            if verbose:
                print('Coverting to int32')
            optimised_df[col] = df[col].astype('int32', errors='ignore')
        elif calculated_col_type == 'float32':
            if verbose:
                print('Coverting to float32')
            optimised_df[col] = df[col].astype('float32', errors='ignore')
        else:
            if verbose:
                print('Copying existing')
            optimised_df[col] = df[col]

    return optimised_df


def get_col_type(df_col: pd.Series):
    """
    Returns a detailed type for a column:
        * date
        * category
        * int8
        * int32
        * float32
        * existing (maintain existing type)
    """
    col_data_type = df_col.dtype
    calculated_col_type = 'existing'

    if col_data_type == 'object':
        # Check whether date or category field
        if is_date_col(df_col):
            calculated_col_type = 'date'
        elif is_category_col(df_col):
            calculated_col_type = 'category'
    elif col_data_type == 'int64' or col_data_type == 'float64':
        if is_int8_col(df_col):
            calculated_col_type = 'int8'
        elif col_data_type == 'int64':
            calculated_col_type = 'int32'
        elif col_data_type == 'float64':
            calculated_col_type = 'float32'

    return calculated_col_type
