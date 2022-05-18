import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from .data_utils import gp_data


class plot_results:
    

    def __init__(self, machine, full_time, X_train, y_train, X_test, y_test, 
    n_train, mean_preds, upper_preds, lower_preds) -> np.ndarray:

        self.machine_name = machine

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
        """GP model training + test data fit"""
        
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
        plt.savefig(
            '{}_interpolation_extrapolation.png'.format(self.machine_name))        
        plt.show()


    def interpolation(self):
        """In-sample fit of GP model"""
        
        plt.figure(figsize=(16, 7))
        
        training = plt.scatter(
            self.X_train, self.y_train, s=[2.75], color='black')
        in_sample = plt.plot(
            self.X_train, self.mean_preds[:self.n_train], marker='.')
        ci = plt.fill_between(
            self.X_train, self.lower_preds[:self.n_train], 
            self.upper_preds[:self.n_train], alpha=0.5, 
            color='darkgrey'
            )

        plt.legend([
        'Interpolation', 'Actual Data', 'Uncertainty: + / - $2 \sigma$', 
        ])

        plt.xlabel('Time', fontsize=14)
        plt.ylim(bottom=-0.1)
        plt.ylabel('kW', fontsize=14)    
        plt.show()


    def performance_deviation_testing(self):
        """
        Find values that lie outside of the confidence region

        Parameters
        ----------
        train_y_inv: training kW values
        time: original training time values
        lower_inv: training lower bound kW values
        upper_inv: training upper bound kW values

        Returns
        -------
        perf_dev_upper: dict of times and values of energy 
        consumption greater than 2 $\sigma$
        perf_dev_lower: dict of times and values of energy 
        consumption less than 2 $\sigma$
        """

        test_preds = self.mean_preds[self.n_train:]
        test_upper = self.upper_preds[self.n_train:]
        test_lower = self.lower_preds[self.n_train:]

         # Upper
        out_of_conf_upper_idx = np.array(
            np.where(self.y_test > test_upper)).flatten()

        out_of_conf_upper_vals = self.y_test[out_of_conf_upper_idx]
        out_of_conf_upper_time = self.X_test[out_of_conf_upper_idx]
        out_of_conf_upper_preds = test_preds[out_of_conf_upper_idx]
        out_of_conf_upper_upper = test_upper[out_of_conf_upper_idx]
        

        perf_dev_upper = {}
        for val, time, pred, upper in zip(
            out_of_conf_upper_vals, out_of_conf_upper_time, 
            out_of_conf_upper_preds, out_of_conf_upper_upper):
            perf_dev_upper[time] = [val, pred, upper]

        # Lower
        out_of_conf_lower_idx = np.array(
            np.where(self.y_test < test_lower)).flatten()
        
        out_of_conf_lower_vals = self.y_test[out_of_conf_lower_idx]
        out_of_conf_lower_time = self.X_test[out_of_conf_lower_idx]
        out_of_conf_preds = test_preds[out_of_conf_lower_idx]
        out_of_conf_lower_lower = test_lower[out_of_conf_lower_idx]
    
        perf_dev_lower = {}
        for val, time, pred, lower in zip(
            out_of_conf_lower_vals, out_of_conf_lower_time, 
            out_of_conf_preds, out_of_conf_lower_lower):
            perf_dev_lower[time] = [val, pred, lower]

        # Testing Registry
        pdd_registry_upper = pd.DataFrame(
            list(perf_dev_upper.items()), columns=['time', 'values']
            )
        pdd_registry_upper['machine'] = 'machine_name'
        pdd_registry_upper['control_limit'] = 'upper'

        pdd_registry_lower = pd.DataFrame(
            list(perf_dev_lower.items()), columns=['time', 'values']
            )
        pdd_registry_lower['machine'] = 'machine_name'
        pdd_registry_lower['control_limit'] = 'lower'

        pdd_registry_testing = pd.concat(
            [pdd_registry_upper, pdd_registry_lower], axis=0).reset_index()
        
        split_df = pd.DataFrame(
            pdd_registry_testing['values'].to_list(),
            columns=['actual_kw', 'expected_kw', 'bound'])

        pdd_registry_testing_final = pd.concat(
            [pdd_registry_testing, split_df], axis=1)
       
        pdd_registry_testing_final.drop('values', axis=1, inplace=True)

        print(pdd_registry_testing_final)


    def load_forecast(self, time_aggregation=int):
        """Next day (24hr) energy (kWh) forecast"""

        test_preds = self.mean_preds[self.n_train:]
        test_upper = self.upper_preds[self.n_train:]
        test_lower = self.lower_preds[self.n_train:]
        
        # Total kW expected to consume with +/-
        if time_aggregation == 30:
            actual_total_energy = np.sum(self.y_test*0.5)
            mean_total_energy = np.sum(test_preds*0.5)
            lower_total_energy = np.sum(test_lower*0.5)
            upper_total_energy = np.sum(test_upper*0.5)
        elif time_aggregation == 10:
            actual_total_energy = np.sum(self.y_test*0.1)
            mean_total_energy = np.sum(test_preds*0.1)
            lower_total_energy = np.sum(test_lower*0.1)
            upper_total_energy = np.sum(test_upper*0.1)
        else:
            raise ValueError(
                'Enter correct time aggregtion for kWh conversion'
                )

        # Plot out-of-sample prediction
        fig, ax = plt.subplots(figsize=(16, 7))
        ax.plot(self.X_test, test_preds, marker='.')
        ax.fill_between(
                self.X_test, test_lower, test_upper, alpha=0.5, 
                color='darkgrey'
                )
        plt.ylabel('kW')
        #plt.title('{} Forecasted Next Day Consumption'.format(machine_name))

        textstr = '\n'.join((
        'kWh = Area Under the Curve',
        r'Upper kWh=%.2f' % (upper_total_energy, ),
        r'Average kWh=%.2f' % (mean_total_energy, ),
        r'Lower kWh=%.2f' % (lower_total_energy, ),
        r'Actual kWh=%.2f' % (actual_total_energy, )))

        props = dict(boxstyle='round', facecolor='white', alpha=0.5)

        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)
        
        fig.savefig(
            '{}_forecast.png'.format(self.machine_name))  
        
        plt.show()


    def quality_control(self):
        """
        ## Within CI (2 stddev) ##

        If truth < upper, then truth - upper = negative value
        If truth > lower, then truth - lower = positive value

        ## Out of CI (2 stddev) ##

        If truth > upper, then truth - upper = positive value
        If truth < lower, then truth - lower = negative value
        """

        ## Training SPC ##

        deviation = self.y_train - self.mean_preds[:self.n_train]
        deviation_upper = self.y_train - self.upper_preds[:self.n_train]
        deviation_lower = self.y_train - self.lower_preds[:self.n_train]

        upper = np.argwhere(deviation_upper > 0.0)
        lower = np.argwhere(deviation_lower < 0.0)

        fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(16, 12))
        
        ax[0].scatter(self.X_train, deviation, alpha=0.5)
        ax[0].scatter(self.X_train[upper], deviation[upper], color='red')
        ax[0].scatter(self.X_train[lower], deviation[lower], color='red')
        ax[0].set_title('Training Energy Performance Deviations')
        ax[0].set_ylabel('kW (difference in predicted vs. actual)')
        ax[0].legend(
            ['Actual kW - Predicted kW', 
            'Upper and Lower Control Limit +/- $2 \sigma$']
            )

        ax[1].scatter(self.X_train, np.cumsum(deviation), alpha=0.5)
        ax[1].plot(
            self.X_train, pd.Series(np.cumsum(deviation)).rolling(6).mean(), 
            color='orange')
        ax[1].plot(
            self.X_train, pd.Series(np.cumsum(deviation)).rolling(60).mean(), 
            color='green')
        ax[1].set_title(
            'Training Energy Performance Cumulative Deviations', size=14)
        ax[1].set_ylabel(
            'kW (Cumulative difference in predicted vs. actual)', size=14)
        ax[1].set_xlabel(
            'Time (Month - Day - Hour)', size=14)
        ax[1].legend(
            ['1hr Moving Average', '6hr Moving Average', 'Cumulative Deviations'])
        
        fig.savefig(
            '{}_training_spc.png'.format(self.machine_name))  

        plt.show()

        ## Testing SPC ##

        test_deviation = self.y_test - self.mean_preds[self.n_train:]
        test_deviation_upper = self.y_test - self.upper_preds[self.n_train:]
        test_deviation_lower = self.y_test - self.lower_preds[self.n_train:]

        test_upper = np.argwhere(test_deviation_upper > 0.0)
        test_lower = np.argwhere(test_deviation_lower < 0.0)

        fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(16, 12))

        ax[0].scatter(self.X_test, test_deviation, alpha=0.5)
        ax[0].scatter(
            self.X_test[test_upper], test_deviation[test_upper], color='red')
        ax[0].scatter(
            self.X_test[test_lower], test_deviation[test_lower], color='red')
        ax[0].set_title('Energy Performance Deviations', size=14)
        ax[0].set_ylabel('kW (difference in predicted vs. actual)', size=14)
        ax[0].legend([
            'Actual kW - Predicted kW', 
            'Upper and Lower Control Limit +/- $2 \sigma$'])

        ax[1].scatter(self.X_test, np.cumsum(test_deviation), alpha=0.5)
        ax[1].plot(
            self.X_test, pd.Series(np.cumsum(test_deviation)).rolling(6).mean(),
            color='orange'
            )
        ax[1].set_title('Energy Performance Cumulative Deviations', size=14)
        ax[1].set_ylabel(
            'kW (Cumulative difference in predicted vs. actual)', size=14
            )
        ax[1].set_xlabel('Time (Month - Day - Hour)', size=14)
        ax[1].legend(['1hr Moving Average', 'Cumulative Deviations'])

        fig.savefig(
            '{}_testing_spc.png'.format(self.machine_name))  
        
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
        machine=machine, full_time=full_time, X_train=train_time, 
        y_train=y_train, X_test=test_time, y_test=y_test, n_train=n_train,
        mean_preds=mean_inv, upper_preds=upper_inv, lower_preds=lower_inv
    )

    plotter.interpolation_extrapolation()
    plotter.interpolation()
    plotter.performance_deviation_testing()
    plotter.load_forecast(time_aggregation=time)
    plotter.quality_control()

if __name__ == "__main__":
    main()