import pandas as pd
from dateutil.parser import parse

def is_date(string):
    try:
        parse(string)
        return True
    except ValueError:
        return False

def optimise_df(df):
    optimised_df = pd.DataFrame()

    for col in df.columns.values:
        col_type = df[col].dtype
        print('Field:', col, 'type:', col_type)

        if col_type == 'object':
            # Check if it's a date
            date_col = False
            top_vals = df[col][:100].values
            for val in top_vals:
                if is_date(val):
                    date_col = True
                    break

            if date_col:
                print('Coverting to date')
                optimised_df[col] = pd.to_datetime(df[col], errors='coerce')
            else:
                # Not a date, so check whether to change value to a category

                num_unique_values = len(df[col].unique())
                num_total_values = len(df[col])
                # Check we will get at least half the values removed by changing to category
                if num_unique_values / num_total_values < 0.5:
                    # fill missing values with NA
                    print('Filling missing values with NA')
                    df[col].fillna('NA', inplace=True)
                    print('Coverting to category')
                    optimised_df[col] = df[col].astype('category')
                else:
                    print('Copying existing')
                    optimised_df[col] = df[col]

        elif col_type == 'int64' or col_type == 'float64':
            # Check if all values are 1 or 0
            int8_mask = optimised_df[col].isin([0, 1])

            if False in int8_mask:
                # Values other than 1 or 0 are present
                if col_type == 'int64':
                    print('Coverting to int32')
                    optimised_df[col] = df[col].astype('int32',  errors='ignore')
                else:
                    print('Coverting to float32')
                    optimised_df[col] = df[col].astype('float32', errors='ignore')
            else:
                # All values are 1 or 0 - can be converted to int8
                optimised_df[col] = df[col].astype('int8',  errors='ignore')
        else:
            print('Copying existing')
            optimised_df[col] = df[col]

    return optimised_df