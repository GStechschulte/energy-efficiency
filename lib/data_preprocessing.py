from lib.util import helper
from sklearn.preprocessing import StandardScaler
import pandas as pd



def preprocess(table, min_train_date, end_train_date, end_test_date):
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
    