import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from .data_utils_v2 import gp_data


class plot_results:

    def __init__(self, full_time, X_train, y_train, X_test, y_test, 
    n_train, mean_preds, upper_preds, lower_preds) -> np.ndarray:

        # Training / testing data
        self.full_time = full_time
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.n_train = n_train

        # Inference data
        self.mean_preds = mean_preds
        self.upper_preds = upper_preds
        self.lower_preds = lower_preds


    def interpolation_extrapolation(self):
        
        plt.figure(figsize=(16, 7))

        full_kw = np.concatenate([self.y_train, self.y_test])
        
        training = plt.scatter(
            self.full_time, full_kw, s=[2.75], color='black')
        in_sample = plt.plot(
            self.X_train, self.mean_preds[:self.n_train], marker='.')
        #testing = plt.scatter(
        #    self.X_test, self.y_test, s=[2.75], color='black')
        out_sample = plt.plot(
            self.X_test, self.mean_preds[self.n_train:], marker='.')
        ci = plt.fill_between(
            self.full_time, self.lower_preds, self.upper_preds, alpha=0.5, 
            color='darkgrey')

        plt.legend([
        'Interpolation', 'Extrapolation', 'Actual Data', 
        'Uncertainty: + / - $2 \sigma$', 
        ])

        plt.xlabel('Time', fontsize=14)
        plt.ylim(bottom=-0.1)
        plt.ylabel('kW', fontsize=14)
        
        #plt.title(
        #    'Machine: {} \n Time Aggregation: {}'.format(
        #        machine_name, time_aggregation))
        
        plt.show()


def main(machine=str, time=int, mean_preds=np, lower_preds=np, upper_preds=np):

    machine_name = machine + '_' + str(time) + 'T'

    # Initialize gp_data query and preprocessing
    data = gp_data(
        machine=machine_name, freq=time, normalize_time=True
    )

    data.preprocessing()
    
    # 1.a) get time data
    full_time, train_time, test_time = data.get_time()

    # 1.b) get kW data
    y_train, y_test, n_train = data.get_orig_values()

    # 1.c) get mean, lower, upper
    mean_inv, lower_inv, upper_inv = data.inverse_transform(
        observed_preds=mean_preds, lower=lower_preds, upper=upper_preds
    )

    # 2.) start calling plotting functions
    plotter = plot_results(
        full_time=full_time, X_train=train_time, y_train=y_train, 
        X_test=test_time, y_test=y_test, n_train=n_train,
        mean_preds=mean_inv, upper_preds=upper_inv, lower_preds=lower_inv
    )

    plotter.interpolation_extrapolation()


if __name__ == "__main__":
    main()