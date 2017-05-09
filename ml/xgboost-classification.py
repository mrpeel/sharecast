
# coding: utf-8

# ## Load and prep columns

import xgboost as xgb
import numpy as np
import pandas as pd
from xgboost.sklearn import XGBClassifier
from xgboost import plot_importance
from matplotlib import pyplot
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from scipy import stats
from datetime import datetime as dt
from dateutil.parser import parse
import time


pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 200)

def load_data:
    # Define columns
    data_columns = ['symbol', 'quoteDate', 'adjustedPrice', 'volume', 'previousClose', 'change', 'changeInPercent', 
                    '52WeekHigh', '52WeekLow', 'changeFrom52WeekHigh', 'changeFrom52WeekLow', 
                    'percebtChangeFrom52WeekHigh', 'percentChangeFrom52WeekLow', 'Price200DayAverage', 
                    'Price52WeekPercChange', '1WeekVolatility', '2WeekVolatility', '4WeekVolatility', '8WeekVolatility', 
                    '12WeekVolatility', '26WeekVolatility','52WeekVolatility','4WeekBollingerPrediction', '4WeekBollingerType',
                    '12WeekBollingerPrediction', '12WeekBollingerType', 'allordpreviousclose', 'allordchange', 
                    'allorddayshigh', 'allorddayslow', 'allordpercebtChangeFrom52WeekHigh', 
                    'allordpercentChangeFrom52WeekLow', 'asxpreviousclose', 'asxchange', 'asxdayshigh', 
                    'asxdayslow', 'asxpercebtChangeFrom52WeekHigh', 'asxpercentChangeFrom52WeekLow', 'exDividendDate', 
                    'exDividendPayout', '640106_A3597525W', 'AINTCOV', 'AverageVolume', 'BookValuePerShareYear', 
                    'CashPerShareYear', 'DPSRecentYear', 'EBITDMargin', 'EPS', 'EPSGrowthRate10Years', 
                    'EPSGrowthRate5Years', 'FIRMMCRT', 'FXRUSD', 'Float', 'GRCPAIAD', 'GRCPAISAD', 'GRCPBCAD', 
                    'GRCPBCSAD', 'GRCPBMAD', 'GRCPNRAD', 'GRCPRCAD', 'H01_GGDPCVGDP', 'H01_GGDPCVGDPFY', 'H05_GLFSEPTPOP', 
                    'IAD', 'LTDebtToEquityQuarter', 'LTDebtToEquityYear', 'MarketCap',
                    'NetIncomeGrowthRate5Years', 'NetProfitMarginPercent', 'OperatingMargin', 'PE',
                    'PriceToBook', 'ReturnOnAssets5Years', 'ReturnOnAssetsTTM', 'ReturnOnAssetsYear', 
                    'ReturnOnEquity5Years', 'ReturnOnEquityTTM', 'ReturnOnEquityYear', 'RevenueGrowthRate10Years', 
                    'RevenueGrowthRate5Years', 'TotalDebtToAssetsQuarter', 'TotalDebtToAssetsYear', 
                    'TotalDebtToEquityQuarter', 'TotalDebtToEquityYear', 'bookValue', 'earningsPerShare', 
                    'ebitda', 'epsEstimateCurrentYear', 'marketCapitalization', 'peRatio', 'pegRatio', 'pricePerBook', 
                    'pricePerEpsEstimateCurrentYear', 'pricePerEpsEstimateNextYear', 'pricePerSales']

    returns = {
        '1': 'Future1WeekReturn',
        '2': 'Future2WeekReturn',
        '4': 'Future4WeekReturn',
        '8': 'Future8WeekReturn',
        '12': 'Future12WeekReturn',
        '26': 'Future26WeekReturn',
        '52': 'Future52WeekReturn',
        '1ra': 'Future1WeekRiskAdjustedReturn',
        '2ra': 'Future2WeekRiskAdjustedReturn',
        '4ra': 'Future4WeekRiskAdjustedReturn',
        '8ra': 'Future8WeekRiskAdjustedReturn',
        '12ra': 'Future12WeekRiskAdjustedReturn',
        '26ra': 'Future26WeekRiskAdjustedReturn',
        '52ra': 'Future52WeekRiskAdjustedReturn'
    }

    # Load data
    raw_data = pd.read_csv('data/companyQuotes-20170417-001.csv')
    raw_data.head(5)


def clip_data:
    # Clip values less than -99 (represents losing all money, can't go below -100)
    columns_to_clip = ['Future1WeekRiskAdjustedReturn', 'Future2WeekRiskAdjustedReturn', 'Future4WeekRiskAdjustedReturn',
                    'Future8WeekRiskAdjustedReturn', 'Future12WeekRiskAdjustedReturn', 'Future26WeekRiskAdjustedReturn',
                    'Future52WeekRiskAdjustedReturn']

    for column in columns_to_clip:
        raw_data[column] = raw_data[column].clip(-99)

def use_all_data:
    filtered_data = raw_data

def use_subset_data:
    # Run filter for a few companies
    include_symbols = ['BHP', 'CBA', 'AOU', 'AYS', 'ATT']
    reduced_data = raw_data[raw_data['symbol'].isin(include_symbols)]
    print(len(reduced_data))

    filtered_data = reduced_data


def bucketise_data:
    # Set-up data buckets
    base_buckets = np.array([3.0, 6.0, 12.5, 25.0, 50.0, 100.0])

    buckets_1 = base_buckets / 52 * 1
    buckets_2 = base_buckets / 52 * 2
    buckets_4 = base_buckets / 52 * 4
    buckets_8 = base_buckets / 52 * 8
    buckets_12 = base_buckets / 52 * 12
    buckets_26 = base_buckets / 52 * 26
    buckets_52 = base_buckets

    buckets = {
        '1': buckets_1,
        '2': buckets_2,
        '4': buckets_4,
        '8': buckets_8,
        '12': buckets_12,
        '26': buckets_26,
        '52': buckets_52,
        '1ra': buckets_1,
        '2ra': buckets_2,
        '4ra': buckets_4,
        '8ra': buckets_8,
        '12ra': buckets_12,
        '26ra': buckets_26,
        '52ra': buckets_52,
    }

    for bucket in buckets:
        buckets[bucket] = np.insert(buckets[bucket], 0, [-99, -25.0, -12.5, 0.0])
        buckets[bucket] = np.append(buckets[bucket], 99999999)
        print(bucket)
        print(buckets[bucket])

    # Add cateogry values to data frames
    group_names = ['Huge loss', 'Serious loss', 'Loss', 'Poor', 'Marginal', 'Average', 'Good', 'Excellent', 'Risky gain',
                'Off the charts gain']


    for key in returns:
        print('-----')
        return_column = returns[key]
        new_column = 'Cat' + return_column
        bins = buckets[key] 
        # bins = buckets['52'] 
        print('Creating', new_column, 'with bins:', bins)
        filtered_data[new_column] = pd.cut(filtered_data[return_column], bins=bins, labels=group_names, include_lowest=True)
        print('New col added')

    filtered_data.head(20)


def plot_bucketised_data:
    # Plot values in categories
    for key in returns:
        print(returns[key])
        return_column = returns[key]
        new_column = 'Cat' + return_column
        ret_data[new_column].value_counts().plot(kind='bar')
        pyplot.show()
    

def plot_return_histograms(data_set):
    # Plot values for each potential target using bucket values
    for key in returns:
        print('-----')
        return_column = returns[key]
        bins = buckets[key] 
        print(return_column)
        data_set.hist(column=return_column,bins=bins)
        pyplot.show()


        print('Instances: ', data_set[return_column].count())
        print('Mean: ', raw_data[return_column].mean())
        print('Min: ', data_set[return_column].min())
        print('25th percentile: ', data_set[return_column].quantile(0.25))
        print('Median: ', data_set[return_column].median())
        print('75th percentile: ', data_set[return_column].quantile(0.75))
        print('Max: ', data_set[return_column].max())
        print('Std deviation: ', data_set[return_column].std())
        print('Variance: ', data_set[return_column].var())
        print('Skew: ', data_set[return_column].skew())


def set_target_column:
    # target_column = returns['8']
    target_column = 'CatFuture12WeekReturn'


def remove_outliers:
    # Check outliers
    outliers = raw_data.loc[(raw_data[target_column] > 100) | (raw_data[target_column] < -50)]
    print(len(outliers))

    exclude_symbols = outliers['symbol'].unique()

    # Remove rows in the excluded symbols list
    filtered_data = raw_data[~raw_data['symbol'].isin(exclude_symbols)]


def remove_missing_y_data:
    # Remove rows missing the target column
    filtered_data = filtered_data.dropna(subset=[target_column], how='all')

    # Create y_data
    base_y_data = filtered_data[target_column].values


    # Filter down data to the X columns being used
    filtered_data = filtered_data[data_columns]


    print(filtered_data.dtypes)

    print('Min:',min(y_data),', Max:', max(y_data))




def is_date(string):
    try: 
        parse(string)
        return True
    except:
        return False

def convert_date_to_ordinal(date_val):
    if(pd.isnull(date_val)):
        return -99999
    
    elif(type(date_val) is str):
        if(is_date(date_val)):
            return parse(date_val).toordinal()
        else:
            return -99999

    elif(type(date_val) is int or type(date_val) is float):
        return date_val

def prepare_and_copy_x_data:
    # Fix date values - convert to ordinals
    filtered_data['quoteDate'] = filtered_data['quoteDate'].apply(lambda x: convert_date_to_ordinal(x))

    # print(filtered_data['exDividendDate'].apply(lambda x: convert_date_to_ordinal(x)))
    filtered_data['exDividendDate'] = filtered_data['exDividendDate'].apply(lambda x: convert_date_to_ordinal(x))

    print(filtered_data.head(5))

    # Convert categorical variables to boolean fields
    #  4WeekBollingerPrediction              
    #  4WeekBollingerType                    
    #  12WeekBollingerPrediction             
    #  12WeekBollingerType                   

    filtered_data = pd.get_dummies(data=filtered_data, columns=['symbol', '4WeekBollingerPrediction', '4WeekBollingerType', 
                                                                '12WeekBollingerPrediction', '12WeekBollingerType'])


    # Fill nan values with placeholder and check for null values
    filtered_data = filtered_data.fillna(-99999)
    print(pd.isnull(filtered_data).any())

    # Check data types
    print(filtered_data.dtypes)

    # Copy over X_data columns
    X_data = filtered_data.values

def prepare_and_copy_y_data:   
    le = preprocessing.LabelEncoder()
    y_data = le.fit_transform(base_y_data)
    print(list(le.classes_))

    # View some sample records
    print(X_data[range(0,5)])

    # View some sample records
    print(y_data[range(0,5)])


# ## Run xgboost classifier

# In[ ]:

from xgboost.sklearn import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import time

# Split into train and test data
print('Splitting data')
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, train_size=0.7, test_size=0.3)

print('Training for', target_column)

# Fit model with training set
start = time.time()
model = xgb.XGBClassifier(nthread=-1)
model.fit(X_train, y_train)
# Output model settings
fit_time = time.time()
print(model)
print('Fit elapsed time: %d' % (fit_time - start))


# make predictions for test data
predictions = model.predict(X_test)
predition_time = time.time()
print('Prediction elapsed time: %d' % (predition_time - fit_time))
print(model)

# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print('Accuracy:', accuracy)

print(confusion_matrix(y_test, predictions))


# In[ ]:

print(le.classes_)
print(confusion_matrix(y_test, predictions))



def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return array[idx]

def grid_search_model:
    print("Work through parameter optimization")
    
    model = xgb.XGBClassifier(nthread=-1)

    kfold = KFold(n_splits=4, shuffle=True)


    print("Set non-optimised baseline")
    round_err = []
    for r in range(0, 3):
        err = []
        for train_index, test_index in kfold.split(X_data):
            model.fit(X_data[train_index],y_data[train_index])
            predictions = model.predict(X_data[test_index])
            actuals = y_data[test_index]
            err.append(accuracy_score(actuals, predictions))

        print(np.mean(err))
        round_err.append(np.mean(err))

    baseline_accuracy = np.mean(round_err)

    print("Average baseline accuracy: %f" % baseline_accuracy)
    print('-----')
    
#     n_estimators=[1000, 1500, 2000]
        
#     param_grid = dict(n_estimators=n_estimators)

#     grid_search = GridSearchCV(model,param_grid, scoring="accuracy", cv=kfold, verbose=2, n_jobs=-1)
#     n_estimators_r = []

#     for r in range(0, 3):
#         grid_result = grid_search.fit(X_data, y_data)
#         # summarize results
#         print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
#         n_estimators_r.append(grid_result.best_params_['n_estimators'])
#         means = grid_result.cv_results_['mean_test_score']
#         stds = grid_result.cv_results_['std_test_score']
#         params = grid_result.cv_results_['params']

#     n_estimators = find_nearest(n_estimators_r, np.mean(n_estimators_r))
    
#     model.n_estimators = n_estimators
    
#     print("Averaged best n_estimators: %f " % n_estimators)
#     print('-----')  

    model.n_estimators = 1500
    
    max_depth = [20, 30, 40, 50]
    param_grid = dict(max_depth=max_depth)

    grid_search = GridSearchCV(model,param_grid, scoring="accuracy", cv=kfold, verbose=2, n_jobs=-1)
    max_depths = []

    for r in range(0, 3):
        grid_result = grid_search.fit(X_data, y_data)
        # summarize results
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        max_depths.append(grid_result.best_params_['max_depth'])
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']

    max_depth = find_nearest(max_depths, np.mean(max_depths))
    
    model.max_depth = max_depth
    
    print("Averaged best max depth: %f " % max_depth)
    print('-----')    
        
    learning_rate = [0.01, 0.05, 0.1, 0.15]
    param_grid = dict(learning_rate=learning_rate)

    grid_search = GridSearchCV(model,param_grid, scoring="accuracy", cv=kfold, verbose=2, n_jobs=-1)
    learning_rates = []

    for r in range(0, 3):
        grid_result = grid_search.fit(X_data, y_data)
        # summarize results
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        learning_rates.append(grid_result.best_params_['learning_rate'])
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']

    learning_rate = find_nearest(learning_rates, np.mean(learning_rates))
    
    model.learning_rate = learning_rate
    
    print("Averaged best learning rate: %f " % learning_rate)
    print('-----')     


    samples = [0.6,0.8,1.0] #[i/100.0 for i in range(60,101, 5)]
    param_grid = dict(subsample=samples)

    grid_search = GridSearchCV(model,param_grid, scoring="accuracy", cv=kfold, verbose=2, n_jobs=-1)
    subsamples = []

    for r in range(0, 3):
        grid_result = grid_search.fit(X_data, y_data)
        # summarize results
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        subsamples.append(grid_result.best_params_['subsample'])
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']

    subsample = find_nearest(subsamples, np.mean(subsamples))
    
    model.subsample = subsample
    
    print("Averaged best subsample: %f " % subsample)
    print('-----')

    param_grid = dict(colsample_bytree=samples)

    grid_search = GridSearchCV(model,param_grid, scoring="accuracy", cv=kfold, verbose=2, n_jobs=-1)
    colsample_bytrees = []

    for r in range(0, 3):
        grid_result = grid_search.fit(X_data, y_data)
        # summarize results
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        colsample_bytrees.append(grid_result.best_params_['colsample_bytree'])
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']

    colsample_bytree = find_nearest(colsample_bytrees, np.mean(colsample_bytrees))
    
    model.colsample_bytree = colsample_bytree
    
    print("Averaged best colsample_bytree: %f " % colsample_bytree)
    print('-----')

    param_grid = dict(colsample_bylevel=samples)

    grid_search = GridSearchCV(model,param_grid, scoring="accuracy", cv=kfold, verbose=2, n_jobs=-1)
    colsample_bylevels = []

    for r in range(0, 3):
        grid_result = grid_search.fit(X_data, y_data)
        # summarize results
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        colsample_bylevels.append(grid_result.best_params_['colsample_bylevel'])
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']

    colsample_bylevel = find_nearest(colsample_bylevels, np.mean(colsample_bylevels))
    
    model.colsample_bylevel = colsample_bylevel
    
    print("Averaged best colsample_bylevel: %f " % colsample_bylevel)
    print('-----')

    # Retest with new parameters
    round_err = []
    for r in range(0, 3):
        for train_index, test_index in kfold.split(X_data):
            xgb_model = xgb.XGBClassifier(nthread=-1, colsample_bytree = colsample_bytree, 
                                         learning_rate = learning_rate, max_depth = max_depth, 
                                         n_estimators = n_estimators, subsample = subsample,
                                         colsample_bylevel = colsample_bylevel)
            xgb_model.fit(X_data[train_index],y_data[train_index])
            predictions = model.predict(X_data[test_index])
            actuals = y_data[test_index]
            err.append(accuracy_score(actuals, predictions))
               
        print(np.mean(err))
        round_err.append(np.mean(err))

    tuned_accuracy = np.mean(round_err)

    print("Average tuned accuracy: %s" % tuned_accuracy)
    improvement = tuned_accuracy - baseline_accuracy
    print('-----')
    print('Optimisation improvement result: %s, %s%%' % (improvement, improvement / baseline_accuracy * 100))
    print('-----')
    print(xgb_model)
    print('-----')


# In[ ]:




# In[ ]:

from sklearn.metrics import mean_absolute_error
import time

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return array[idx]

if __name__ == "__main__":
    print("Work through parameter optimization")
    
    model = xgb.XGBClassifier(nthread=-1)

    kfold = KFold(n_splits=5, shuffle=True)


    print("Set non-optimised baseline")
    round_err = []
    for r in range(0, 5):
        err = []
        for train_index, test_index in kfold.split(X_data):
            model.fit(X_data[train_index],y_data[train_index])
            predictions = model.predict(X_data[test_index])
            actuals = y_data[test_index]
            err.append(accuracy_score(actuals, predictions))

        print(np.mean(err))
        round_err.append(np.mean(err))

    baseline_accuracy = np.mean(round_err)

    print("Average baseline accuracy: %f" % baseline_accuracy)
    print('-----')

    max_depth = [10, 30, 50]
    param_grid = dict(max_depth=max_depth)

    grid_search = GridSearchCV(model,param_grid, scoring="accuracy", cv=kfold, verbose=1, n_jobs=-1)
    max_depths = []

    for r in range(0, 5):
        grid_result = grid_search.fit(X_data, y_data)
        # summarize results
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        max_depths.append(grid_result.best_params_['max_depth'])
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']

    max_depth = find_nearest(max_depths, np.mean(max_depths))
    
    model.max_depth = max_depth
    
    print("Averaged best max depth: %f " % max_depth)
    print('-----')    
    n_estimators=[500, 2500, 4500, 6500, 8500]
        
    param_grid = dict(n_estimators=n_estimators)

    grid_search = GridSearchCV(model,param_grid, scoring="accuracy", cv=kfold, verbose=1, n_jobs=-1)
    n_estimators_r = []

    for r in range(0, 5):
        grid_result = grid_search.fit(X_data, y_data)
        # summarize results
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        n_estimators_r.append(grid_result.best_params_['n_estimators'])
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']

    n_estimators = find_nearest(n_estimators_r, np.mean(n_estimators_r))
    
    model.n_estimators = n_estimators
    
    print("Averaged best n_estimators: %f " % n_estimators)
    print('-----')  
        
    learning_rate = [0.01, 0.05, 0.1, 0.15, 0.2]
    param_grid = dict(learning_rate=learning_rate)

    grid_search = GridSearchCV(model,param_grid, scoring="accuracy", cv=kfold, verbose=1, n_jobs=-1)
    learning_rates = []

    for r in range(0, 5):
        grid_result = grid_search.fit(X_data, y_data)
        # summarize results
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        learning_rates.append(grid_result.best_params_['learning_rate'])
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']

    learning_rate = find_nearest(learning_rates, np.mean(learning_rates))
    
    model.learning_rate = learning_rate
    
    print("Averaged best learning rate: %f " % learning_rate)
    print('-----')     


    samples = [0.6,0.8,1.0] #[i/100.0 for i in range(60,101, 5)]
    param_grid = dict(subsample=samples)

    grid_search = GridSearchCV(model,param_grid, scoring="accuracy", cv=kfold, verbose=1, n_jobs=-1)
    subsamples = []

    for r in range(0, 5):
        grid_result = grid_search.fit(X_data, y_data)
        # summarize results
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        subsamples.append(grid_result.best_params_['subsample'])
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']

    subsample = find_nearest(subsamples, np.mean(subsamples))
    
    model.subsample = subsample
    
    print("Averaged best subsample: %f " % subsample)
    print('-----')

    param_grid = dict(colsample_bytree=samples)

    grid_search = GridSearchCV(model,param_grid, scoring="accuracy", cv=kfold, verbose=1, n_jobs=-1)
    colsample_bytrees = []

    for r in range(0, 5):
        grid_result = grid_search.fit(X_data, y_data)
        # summarize results
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        colsample_bytrees.append(grid_result.best_params_['colsample_bytree'])
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']

    colsample_bytree = find_nearest(colsample_bytrees, np.mean(colsample_bytrees))
    
    model.colsample_bytree = colsample_bytree
    
    print("Averaged best colsample_bytree: %f " % colsample_bytree)
    print('-----')

    param_grid = dict(colsample_bylevel=samples)

    grid_search = GridSearchCV(model,param_grid, scoring="accuracy", cv=kfold, verbose=1, n_jobs=-1)
    colsample_bylevels = []

    for r in range(0, 5):
        grid_result = grid_search.fit(X_data, y_data)
        # summarize results
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        colsample_bylevels.append(grid_result.best_params_['colsample_bylevel'])
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']

    colsample_bylevel = find_nearest(colsample_bylevels, np.mean(colsample_bylevels))
    
    model.colsample_bylevel = colsample_bylevel
    
    print("Averaged best colsample_bylevel: %f " % colsample_bylevel)
    print('-----')

    # Retest with new parameters
    round_err = []
    for r in range(0, 5):
        err = []
        for train_index, test_index in kfold.split(X_data):
            xgb_model = xgb.XGBClassifier(nthread=-1, colsample_bytree = colsample_bytree, 
                                         learning_rate = learning_rate, max_depth = max_depth, 
                                         n_estimators = n_estimators, subsample = subsample,
                                         colsample_bylevel = colsample_bylevel)
            xgb_model.fit(X_data[train_index],y_data[train_index])
            predictions = model.predict(X_data[test_index])
            actuals = y_data[test_index]
            err.append(accuracy_score(actuals, predictions))
               
        print(np.mean(err))
        round_err.append(np.mean(err))

    tuned_accuracy = np.mean(round_err)

    print("Average tuned accuracy: %s" % tuned_accuracy)
    improvement = tuned_accuracy - baseline_accuracy
    print('-----')
    print('Optimisation improvement result: %s, %s%%' % (improvement, improvement / baseline_accuracy * 100))
    print('-----')
    print(xgb_model)
    print('-----')


# ## Secondary parameters

# In[ ]:

weights = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


#     gamma = [0]
#     param_grid = dict(gamma=gamma)

#     grid_search = GridSearchCV(model, param_grid, scoring="accuracy", cv=kfold, verbose=1, n_jobs=-1)
#     gammas = []

#     for r in range(0, 5):
#         grid_result = grid_search.fit(X_data, y_data)
#         # summarize results
#         print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
#         gammas.append(grid_result.best_params_['gamma'])
#         means = grid_result.cv_results_['mean_test_score']
#         stds = grid_result.cv_results_['std_test_score']
#         params = grid_result.cv_results_['params']

#     gamma = find_nearest(gammas, np.mean(gammas))

#     model.gamma = gamma

#     print("Averaged best gamma: %f " % gamma)
#     print('-----')    

#     min_child_weight = [0]
#     param_grid = dict(min_child_weight=min_child_weight)

#     grid_search = GridSearchCV(model,param_grid, scoring="accuracy", cv=kfold, verbose=1, n_jobs=-1)
#     min_child_weights = []

#     for r in range(0, 5):
#         grid_result = grid_search.fit(X_data, y_data)
#         # summarize results
#         print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
#         min_child_weights.append(grid_result.best_params_['min_child_weight'])
#         means = grid_result.cv_results_['mean_test_score']
#         stds = grid_result.cv_results_['std_test_score']
#         params = grid_result.cv_results_['params']

#     min_child_weight = find_nearest(min_child_weights, np.mean(min_child_weights))

#     model.min_child_weight = min_child_weight

#     print("Averaged best min_child_weight: %f " % min_child_weight)
#     print('-----')

gamma = 0
min_child_weight = 0

reg_lambda = [0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
param_grid = dict(reg_lambda=reg_lambda)

grid_search = GridSearchCV(model,param_grid, scoring="accuracy", cv=kfold, verbose=1, n_jobs=-1)
reg_lambdas = []

for r in range(0, 5):
    grid_result = grid_search.fit(X_data, y_data)
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    reg_lambdas.append(grid_result.best_params_['reg_lambda'])
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']

reg_lambda = find_nearest(reg_lambdas, np.mean(reg_lambdas))

model.reg_lambda = reg_lambda

print("Averaged best reg_lambda: %f " % reg_lambda)
print('-----')

scale_pos_weight = [0, 1, 2, 3, 4, 5]
param_grid = dict(scale_pos_weight=scale_pos_weight)

grid_search = GridSearchCV(model,param_grid, scoring="accuracy", cv=kfold, verbose=1, n_jobs=-1)
scale_pos_weights = []

for r in range(0, 5):
    grid_result = grid_search.fit(X_data, y_data)
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    scale_pos_weights.append(grid_result.best_params_['scale_pos_weight'])
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']

scale_pos_weight = find_nearest(scale_pos_weights, np.mean(scale_pos_weights))

model.scale_pos_weight = scale_pos_weight

print("Averaged best scale_pos_weight: %f " % scale_pos_weight)
print('-----')


reg_alpha = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
param_grid = dict(reg_alpha=reg_alpha)

grid_search = GridSearchCV(model,param_grid, scoring="accuracy", cv=kfold, verbose=1, n_jobs=-1)
reg_alphas = []

for r in range(0, 5):
    grid_result = grid_search.fit(X_data, y_data)
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    reg_alphas.append(grid_result.best_params_['reg_alpha'])
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']

reg_alpha = find_nearest(reg_alphas, np.mean(reg_alphas))

model.reg_alpha = reg_alpha

print("Averaged best reg_alpha: %f " % reg_alpha)
print('-----')
    
base_score = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
param_grid = dict(base_score=base_score)

grid_search = GridSearchCV(model,param_grid, scoring="accuracy", cv=kfold, verbose=1, n_jobs=-1)
base_scores = []

for r in range(0, 5):
    grid_result = grid_search.fit(X_data, y_data)
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    base_scores.append(grid_result.best_params_['base_score'])
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']

base_score = find_nearest(base_scores, np.mean(base_scores))

model.base_score = base_score

print("Averaged best base_score: %f " % base_score)
print('-----')


# Retest with new parameters
round_err = []
for r in range(0, 5):
    err = []
    for train_index, test_index in kfold.split(X_data):
        xgb_model = xgb.XGBRegressor(nthread=-1, colsample_bytree = colsample_bytree, gamma=gamma, 
                                     learning_rate = learning_rate, max_depth = max_depth, 
                                     n_estimators = n_estimators, subsample = subsample,
                                     colsample_bylevel = colsample_bylevel, base_score = base_score,
                                     reg_alpha = reg_alpha, scale_pos_weight = scale_pos_weight,
                                     reg_lambda = reg_lambda, min_child_weight = min_child_weight)
        xgb_model.fit(X_data[train_index],y_data[train_index])
        predictions = model.predict(X_data[test_index])
        actuals = y_data[test_index]
        err.append(accuracy_score(actuals, predictions))
           
    print(np.mean(err))
    round_err.append(np.mean(err))

tuned_error = np.mean(round_err)

print("Average tuned error: %s" % tuned_error)
improvement = baseline_error - tuned_error
print('-----')
print('Optimisation improvement result: %s, %s%%' % (improvement, improvement / baseline_error * 100))
print('-----')
print(xgb_model)
print('-----')


# In[ ]:

model.max_depth = 70
n_estimators=[7000, 7500, 8000, 8500, 9000, 9500, 10000]
    
param_grid = dict(n_estimators=n_estimators)

grid_search = GridSearchCV(model,param_grid, scoring="accuracy", cv=kfold, verbose=1, n_jobs=-1)
n_estimators_r = []

for r in range(0, 5):
    grid_result = grid_search.fit(X_data, y_data)
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    n_estimators_r.append(grid_result.best_params_['n_estimators'])
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']

n_estimators = find_nearest(n_estimators_r, np.mean(n_estimators_r))

model.n_estimators = n_estimators

print("Averaged best n_estimators: %f " % n_estimators)
print('-----')  


# ## Compare model to baseline

# In[ ]:

import time

# Test with base parameters
print('-----')
print('Base model')

base_errs = []
for r in range(0, 5):
    err = []
    for train_index, test_index in kfold.split(X_data):
        start = time.time()
        base_model = xgb.XGBRegressor(nthread=-1)
        base_model.fit(X_data[train_index],y_data[train_index])
        fit_time = time.time()
        predictions = base_model.predict(X_data[test_index])
        prediction_time = time.time()
        actuals = y_data[test_index]
        err.append(accuracy_score(actuals, predictions))
               
    print(np.mean(err))
    base_errs.append(np.mean(err))
    print('Fit elapsed time: %d, Prediction elapsed time: %d' % (fit_time - start, prediction_time - fit_time))

base_error = np.mean(base_errs)

print('-----')
print(base_model)
print("Average base error: %s" % base_error)


# Retest with new parameters
print('-----')
print('Optimised model')

opt_err = []
for r in range(0, 5):
    err = []
    for train_index, test_index in kfold.split(X_data):
        start = time.time()
        tuned_model = xgb.XGBRegressor(base_score=0.35, colsample_bylevel=0.8, colsample_bytree=0.8, 
                                     gamma=0, learning_rate=0.075, max_delta_step=0, max_depth=70, 
                                     min_child_weight=0, missing=None, n_estimators=9500, nthread=-1, 
                                     reg_alpha=0.4, reg_lambda=0.3, scale_pos_weight=0, subsample=0.8)
        tuned_model.fit(X_data[train_index],y_data[train_index])
        fit_time = time.time()
        predictions = tuned_model.predict(X_data[test_index])
        prediction_time = time.time()
        actuals = y_data[test_index]
        err.append(accuracy_score(actuals, predictions))
               
    print(np.mean(err))
    opt_err.append(np.mean(err))
    print('Fit elapsed time: %d, Prediction elapsed time: %d' % (fit_time - start, prediction_time - fit_time))


tuned_error = np.mean(opt_err)

print('-----')
print(tuned_model)
print("Average tuned error: %s" % tuned_error)
improvement = base_error - tuned_error
print('-----')
print('Optimisation improvement result: %s, %s%%' % (improvement, improvement / base_error * 100))
print('-----')



# ## Data checks

# In[ ]:

# Check correlations 
filtered_data[data_columns].corr()

