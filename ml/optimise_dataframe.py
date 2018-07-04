import pandas as pd
from dateutil.parser import parse


def is_date(string):
    """
        Checks whether a string is a date by attempting to parse it.
    """
    try:
        parse(string)
        return True
    except:
        return False


def is_date_col(df_col):
    """
        Checks whether the first 100 values in a column convert to dates.
        If yes, returns that this is a date column.
    """
    date_col = False
    # Retrieve first 100 values to check whether dates
    top_vals = df_col[:100].values
    for val in top_vals:
        if is_date(val):
            date_col = True
            break

    return date_col


def is_category_col(df_col):
    """
        Checks whether an object (string) column should be converted to a category.
        If on average there are two values for every category, column should be
        a category.
    """
    num_unique_values = len(df_col.unique())
    num_total_values = len(df_col)
    # Check we will get at least half the values removed by changing to category
    return bool((num_unique_values / num_total_values) < 0.5)


def is_int8_col(df_col):
    """ Checks a numeric column only contains 0s and 1s """
    return df_col.isin([0, 1]).all()


def optimise_df(df):
    """
        Works through the columns of a dataframe and optimises for memory where
        possible.
    """

    optimised_df = pd.DataFrame()

    for col in df.columns.values:
        print('Field:', col, 'type:', df[col].dtype)
        calulated_col_type = get_col_type(df[col])

        if calulated_col_type == 'date':
            print('Coverting to date')
            optimised_df[col] = pd.to_datetime(df[col], errors='coerce')
        elif calulated_col_type == 'category':
            # fill missing values with NA
            print('Filling missing values with NA')
            df[col].fillna('NA', inplace=True)
            print('Coverting to category')
            optimised_df[col] = df[col].astype('category')
        elif calulated_col_type == 'int8':
            print('Coverting to int8')
            optimised_df[col] = df[col].astype('int8',  errors='ignore')
        elif calulated_col_type == 'int32':
            print('Coverting to int32')
            optimised_df[col] = df[col].astype('int32',  errors='ignore')
        elif calulated_col_type == 'float32':
            print('Coverting to float32')
            optimised_df[col] = df[col].astype('float32', errors='ignore')
        else:
            print('Copying existing')
            optimised_df[col] = df[col]

    return optimised_df


def get_col_type(df_col):
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
