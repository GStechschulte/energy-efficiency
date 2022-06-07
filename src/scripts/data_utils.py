from . import db_helper
import pandas as pd
import numpy as np
import re
import torch


class gp_data:


    def __init__(self, machine, freq, normalize_time=bool,
    custom_dates=False):

        self.machine = machine
        self.freq = freq
        self.normalize_time = normalize_time
        self.custom_dates = custom_dates


    def get_time(self):
        
        return self.original_time, self.time_vals_train, self.time_vals_test
    

    def get_orig_values(self):

        return self.y_train.numpy(), self.y_test.numpy(), self.n_train
   
    
    def get_preprocessed(self):
        
        return self.X_train_norm, self.y_train_scale, self.X_test_norm, \
            self.y_test_scale, self.n_train

    
    def query_data(self):
        
        # Query table production weekday time series
        if bool(re.findall('trockner', self.machine)) \
            or bool(re.findall('uv_sigma_line', self.machine)):
            df = db_helper.query_table(table=self.machine)  

            return df
        
        else:
            df = db_helper.weekday_time_series(sensor_id=self.machine)
            
            return df


    def preprocessing(self):

        df = self.query_data()

        # Convert negative kW values to 0.0
        df['kw'] = df['kw'].apply(lambda x: 0.0 if x == -0.0 else x)

        if self.normalize_time == True and self.custom_dates == True:
            # Normalize integer based time encodings
            df = df[df.index <= '2021-10-15 12:00:00']
            self.original_time = pd.DatetimeIndex(df.index)
            time_int_range = np.arange(0, len(df)*self.freq, self.freq)
            df['t'] = time_int_range
            self.x_max, self.x_min = df['t'].max(), df['t'].min()
            df['t'] = (
                (df['t'] - df['t'].min()) / (df['t'].max() - df['t'].min()))

        elif self.normalize_time == True and self.custom_dates == False:
            self.original_time = pd.DatetimeIndex(df.index)
            time_int_range = np.arange(0, len(df)*self.freq, self.freq)
            df['t'] = time_int_range
            self.x_max, self.x_min = df['t'].max(), df['t'].min()
            df['t'] = (
                (df['t'] - df['t'].min()) / (df['t'].max() - df['t'].min()))
        else:
            raise ValueError(
                'Must pass either a valid data scaling method or custom dates'
                )

        if self.custom_dates == True:
            self.n_train = self.original_time.get_loc('2021-10-15 00:00:00')
            training_window = df[df.index < '2021-10-15 00:00:00']
            testing_window = df[
                (df.index >= '2021-10-15 00:00:00') & \
                    (df.index <= '2021-10-15 12:00:00')
                ]

            # Training
            self.X_train_norm = torch.from_numpy(
                training_window['t'].values).to(torch.float64)
            self.y_train = torch.from_numpy(
                training_window['kw'].values).to(torch.float64)
            self.time_vals_train = training_window.index.to_list()

            # Testing
            self.X_test_norm = torch.from_numpy(
                df['t'].values).to(torch.float64)
            self.y_test = torch.from_numpy(
                testing_window['kw'].values).to(torch.float64)
            self.time_vals_test = testing_window.index.to_list()

            self.y_train_mean = torch.mean(self.y_train)
            self.y_train_std = torch.std(self.y_train)
            
            # Standardizing helps with hyperparameter initialization
            self.y_train_scale = (
                (self.y_train - self.y_train_mean) / (self.y_train_std))
            self.y_test_scale = (
                (self.y_test - self.y_train_mean) / (self.y_train_std))

        else:
            X = df['t'].values
            y = df['kw'].values
            n = len(X)
            prop_train = 0.8
            self.n_train = round(prop_train * n)

            # Training
            self.X_train_norm = torch.from_numpy(
                X[:self.n_train]).to(torch.float64)
            self.y_train = torch.from_numpy(
                y[:self.n_train]).to(torch.float64)
            self.time_vals_train = self.original_time[:self.n_train]

            # Testing
            self.X_test_norm = torch.from_numpy(X).to(torch.float64)
            self.y_test = torch.from_numpy(y[self.n_train:]).to(torch.float64)
            self.time_vals_test = self.original_time[self.n_train:]

            # Standardizing helps with hyperparameter initialization
            self.y_train_mean = torch.mean(self.y_train)
            self.y_train_std = torch.std(self.y_train)

            self.y_train_scale = (
                (self.y_train - self.y_train_mean) / (self.y_train_std))
            self.y_test_scale = (
                (self.y_test - self.y_train_mean) / (self.y_train_std))


    def inverse_transform(self, observed_preds, lower, upper):
        """
        Using the scale and location of the training data to
        transform the preds and training / test data back to 
        their original scale (kW)

        Parameters
        ----------
        train_y: observed train_y points
        test_y: test_y points
        observed_preds: model predictions
        lower: -2 $\sigma$
        uppwer: +2 $\sigma$

        Returns
        -------
        Inverse transformed parameters
        """

        # Observed preds inverse transform
        observed_preds = observed_preds * self.y_train_std
        observed_preds += self.y_train_mean

        # Confidence region inverse transform
        lower *= self.y_train_std
        upper *= self.y_train_std
        lower += self.y_train_mean
        upper += self.y_train_mean

        return observed_preds.numpy(), lower.numpy(), upper.numpy()