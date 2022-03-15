from lib.util import helper
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import re
import torch


y_train_mean = None
y_train_std = None
x_min = None
x_max = None
original_time = None
time_vals_train = None
time_vals_test = None


def gam_preprocess(table, min_train_date, end_train_date, end_test_date):
    """
    This function. . .
    
    Parameters
    ----------

    Returns
    -------
    """

    # Query table production weekday time series 
    df = helper.weekday_time_series(table)

    # Convert negative kW values to 0.0
    df['kw'] = df['kw'].apply(lambda x: 0.0 if x == -0.0 else x)

    # Subset according to date args.
    min_train_date = pd.Timestamp(min_train_date)
    end_train_date = pd.Timestamp(end_train_date)
    end_test_date = pd.Timestamp(end_test_date)

    train = df[(df.index.date >= min_train_date) & (df.index.date <= end_train_date)]
    test = df[(df.index.date > end_train_date) & (df.index.date <= end_test_date)]

    # Only select target and feature variable
    train_set = pd.DataFrame(data=train['kw'], index=train.index)
    train_set.reset_index(inplace=True)

    test_set = pd.DataFrame(data=test['kw'], index=test.index)
    test_set.reset_index(inplace=True)
    
    # Standardize training and test data using training data: 0 mean and unit variance
    scale = StandardScaler()

    train_set['kw'] = scale.fit_transform(train_set['kw'].values.reshape(-1, 1))
    scale.fit(train_set['kw'].values.reshape(-1, 1))
    test_set['kw'] = scale.transform(test_set['kw'].values.reshape(-1, 1))

    # Rename columns to Prophet conventions
    train_set = train_set.rename(columns={'t': 'ds', 'kw': 'y'})
    test_set = test_set.rename(columns={'t': 'ds', 'kw': 'y'})

    return train_set, test_set

def gp_preprocess(machine, freq, normalize_time=bool, custom_dates=False):

    global y_train_mean
    global y_train_std
    global x_min
    global x_max
    global original_time
    global time_vals_train
    global time_vals_test

    # Query table production weekday time series

    if bool(re.findall('trockner', machine)):
        df = helper.query_table(table=machine)  
    else:
        df = helper.weekday_time_series(sensor_id=machine)

    # Convert negative kW values to 0.0
    df['kw'] = df['kw'].apply(lambda x: 0.0 if x == -0.0 else x)

    # Infer frequency value for normalization
    #freq = pd.infer_freq(df.index)
    original_time = list(df.index)
    time_int_range = np.arange(0, len(df)*freq, freq)
    df['t'] = time_int_range

    if normalize_time == True:
        # Normalize integer based time encodings
        x_max, x_min = df['t'].max(), df['t'].min()
        df['t'] = (df['t'] - df['t'].min()) / (df['t'].max() - df['t'].min())
        X = df['t'].values
    else:
        X = df['t'].values
    
    y = df['kw'].values

    if custom_dates == True:
        print('yo')
    else:
        n = len(X)
        prop_train = 0.8
        n_train = round(prop_train * n)

        # Training
        X_train = torch.from_numpy(X[:n_train]).to(torch.float64)
        y_train = torch.from_numpy(y[:n_train]).to(torch.float64)
        time_vals_train = original_time[:n_train]

        # Testing
        X_test = torch.from_numpy(X).to(torch.float64)
        y_test = torch.from_numpy(y[n_train:]).to(torch.float64)
        time_vals_test = original_time[n_train:]

        # Standardizing helps with hyperparameter initialization
        y_train_mean = torch.mean(y_train)
        y_train_std = torch.std(y_train)

        y_train = (y_train - y_train_mean) / (y_train_std)
        y_test = (y_test - y_train_mean) / (y_train_std)

        return X_train, y_train, X_test, y_test, n_train

def gp_inverse_transform(train_y, test_y, observed_preds, lower, upper):
    """
    . . .
    """

    # Target Variable inverse transform
    train_y *= y_train_std
    train_y += y_train_mean

    test_y *= y_train_std
    test_y += y_train_mean

    # Observed preds inverse transform
    observed_preds = observed_preds.mean * y_train_std
    observed_preds += y_train_mean

    # Confidence region inverse transform
    lower *= y_train_std
    upper *= y_train_std
    lower += y_train_mean
    upper += y_train_mean

    # Time Variable
    #train_x -= x_min
    #scaled = normed_x ( max - min) + min

    return train_y, test_y, observed_preds, lower, upper, original_time, \
        time_vals_train, time_vals_test



