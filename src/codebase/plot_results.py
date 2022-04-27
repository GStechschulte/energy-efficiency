import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


# variable names
orig_time
orig_time_train
orig_time_test

train_y_inv
test_y_inv

observed_preds_inv
lower_inv
upper_inv

machine_name
time_aggregation

def interpolation_extrapolation():
    
    f, ax = plt.subplots(figsize=(16, 7))
    training = ax.scatter(orig_time_train, train_y_inv.numpy(), s=[2.75], color='black')
    in_sample = ax.plot(
        orig_time_train, observed_preds_inv[:n_train].numpy(), marker='.')
    testing = ax.scatter(orig_time_test, test_y_inv.numpy(), s=[2.75], color='black')
    out_sample = ax.plot(
        orig_time_test, observed_preds_inv[n_train:].numpy(), marker='.')
    
    ci = ax.fill_between(
        orig_time, lower_inv.numpy(), upper_inv.numpy(), alpha=0.5, color='darkgrey'
    )

    ax.legend([
    'Interpolation', 'Extrapolation', 'Actual Data', 'Uncertainty: + / - $2 \sigma$'
    ])

    plt.xlabel('Time', fontsize=14)
    plt.ylim(bottom=-0.1)
    plt.ylabel('kW', fontsize=14)
    plt.title(
        'Machine: {} \n Time Aggregation: {}'.format(
            machine_name, time_aggregation))
    plt.show()


def interpolation():

    f, ax = plt.subplots(figsize=(16, 7))
    training = ax.scatter(orig_time_train, train_y_inv.numpy(), s=[2.75], color='black')
    in_sample = ax.plot(
        orig_time_train, observed_preds_inv[:n_train].numpy(),
        marker='.')

    ci = ax.fill_between(
        orig_time_train, lower_inv[:n_train].numpy(), upper_inv[:n_train].numpy(), alpha=0.5, 
        color='darkgrey'
        )

    """
    perf_dev_upper, perf_dev_lower = performance_deviation_training(
    train_y_inv, observed_preds_inv[:n_train], orig_time_train, lower_inv[:n_train], 
    upper_inv[:n_train]
    )
    
    for time, val in perf_dev_upper.items():
        ax.annotate(
            round(val, 2), (time, val), fontsize=12)

    for time, val in perf_dev_lower.items():
        ax.annotate(
            round(val, 2), (time, val), fontsize=12)

    ax.legend([
        'Predictions', 'Observed Data', 'Uncertainty: + / - $2 \sigma$', 'Upper Dev.', 
        'Lower Dev.'
        ])
    """

    plt.xlabel('Time', fontsize=14)
    plt.xticks()
    plt.ylabel('kW', fontsize=14)
    plt.title(
        'Machine: {} \n Time Aggregation: {}'.format(
            machine_name, time_aggregation))
    plt.show()


def performance_deviation():
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
    perf_dev_upper: dict of times and values of energy consumption greater than 
    2 $\sigma$
    perf_dev_lower: dict of times and values of energy consumption less than 
    2 $\sigma$
    """

    test_time = np.array(test_time)

    #ground_truth = np.concatenate([train_y_inv.numpy(), test_y_inv.numpy()])
    ground_truth = test_y_inv.numpy()
    test_preds = test_preds.numpy()

    # Upper
    out_of_conf_upper_idx = np.array(
        np.where(ground_truth > test_upper.numpy())).flatten()

    out_of_conf_upper_vals = ground_truth[out_of_conf_upper_idx]
    out_of_conf_upper_time = test_time[out_of_conf_upper_idx]
    out_of_conf_upper_preds = test_preds[out_of_conf_upper_idx]
    out_of_conf_upper_upper = test_upper.numpy()[out_of_conf_upper_idx]
    

    perf_dev_upper = {}
    for val, time, pred, upper in zip(
        out_of_conf_upper_vals, out_of_conf_upper_time, 
        out_of_conf_upper_preds, out_of_conf_upper_upper):
        perf_dev_upper[time] = [val, pred, upper]

    # Lower
    out_of_conf_lower_idx = np.array(
        np.where(ground_truth < test_lower.numpy())).flatten()
    
    out_of_conf_lower_vals = ground_truth[out_of_conf_lower_idx]
    out_of_conf_lower_time = test_time[out_of_conf_lower_idx]
    out_of_conf_preds = test_preds[out_of_conf_lower_idx]
    out_of_conf_lower_lower = test_lower.numpy()[out_of_conf_lower_idx]
 
    perf_dev_lower = {}
    for val, time, pred, lower in zip(
        out_of_conf_lower_vals, out_of_conf_lower_time, 
        out_of_conf_preds, out_of_conf_lower_lower):
        perf_dev_lower[time] = [val, pred, lower]

    # Testing Registry
    pdd_registry_upper = pd.DataFrame(list(perf_dev_upper.items()), columns=['time', 'values'])
    pdd_registry_upper['machine'] = machine_name
    pdd_registry_upper['control_limit'] = 'upper'

    pdd_registry_lower = pd.DataFrame(list(perf_dev_lower.items()), columns=['time', 'values'])
    pdd_registry_lower['machine'] = machine_name
    pdd_registry_lower['control_limit'] = 'lower'

    pdd_registry_testing = pd.concat([pdd_registry_upper, pdd_registry_lower], axis=0).reset_index()
    
    split_df = pd.DataFrame(
        pdd_registry_testing['values'].to_list(),
        columns=['actual_kw', 'expected_kw', 'bound'])

    pdd_registry_testing_final = pd.concat([pdd_registry_testing, split_df], axis=1)
    pdd_registry_testing_final.drop('values', axis=1, inplace=True)

    print(pdd_registry_testing_final)


def forecasted_consumption():
    """
    Communicates the predicted electricity consumption for the next
    day with uncertainties

    Parameters
    ----------

    Returns
    -------

    """

    # Total kW expected to consume with +/-
    if time_aggregation == '30T':
        mean_total_energy = torch.sum(preds_mean*0.5).numpy()
        lower_total_energy = torch.sum(lower_inv*0.5).numpy()
        upper_total_energy = torch.sum(upper_inv*0.5).numpy()
    elif time_aggregation == '10T':
        mean_total_energy = torch.sum(preds_mean*0.1).numpy()
        lower_total_energy = torch.sum(lower_inv*0.1).numpy()
        upper_total_energy = torch.sum(upper_inv*0.1).numpy()
    else:
        raise ValueError('Code kWh conversion for lower time aggregation')
        

    # Plot out-of-sample prediction
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.plot(test_time, preds_mean.numpy(), marker='.')
    ax.fill_between(
            test_time, lower_inv.numpy(), upper_inv.numpy(), alpha=0.5, 
            color='darkgrey'
            )
    plt.ylabel('kW')
    plt.title('{} Forecasted Next Day Consumption'.format(machine_name))

    textstr = '\n'.join((
    'kWh = Area Under the Curve',
    r'Upper kWh=%.2f' % (upper_total_energy, ),
    r'Average kWh=%.2f' % (mean_total_energy, ),
    r'Lower kWh=%.2f' % (lower_total_energy, )))

    props = dict(boxstyle='round', facecolor='white', alpha=0.5)

    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)


def control_charts():
    """
    ## Within CI (2 stddev) ##

    If truth < upper, then truth - upper = negative value
    If truth > lower, then truth - lower = positive value

    ## Out of CI (2 stddev) ##

    If truth > upper, then truth - upper = positive value
    If truth < lower, then truth - lower = negative value
    """

    performance_deviation(
        test_y_inv=test_truth, test_preds=test_preds, test_time=test_time,
        test_lower=test_lower, test_upper=test_upper)

    deviation = ground_truth.numpy() - preds_mean.numpy()
    deviation_upper = ground_truth.numpy() - upper.numpy()
    deviation_lower = ground_truth.numpy() - lower.numpy()

    upper = np.argwhere(deviation_upper > 0.0)
    lower = np.argwhere(deviation_lower < 0.0)

    fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(16, 12))
    
    ax[0].scatter(orig_time, deviation, alpha=0.5)
    ax[0].scatter(orig_time[upper], deviation[upper], color='red')
    ax[0].scatter(orig_time[lower], deviation[lower], color='red')
    ax[0].set_title('Training Energy Performance Deviations')
    ax[0].set_ylabel('kW (difference in predicted vs. actual)')
    ax[0].legend([
        'Actual kW - Predicted kW', 'Upper and Lower Control Limit +/- $2 \sigma$'])

    ax[1].scatter(orig_time, np.cumsum(deviation), alpha=0.5)
    ax[1].plot(orig_time, pd.Series(np.cumsum(deviation)).rolling(6).mean(), color='orange')
    ax[1].plot(orig_time, pd.Series(np.cumsum(deviation)).rolling(60).mean(), color='green')
    ax[1].set_title('Training Energy Performance Cumulative Deviations', size=14)
    ax[1].set_ylabel('kW (Cumulative difference in predicted vs. actual)', size=14)
    ax[1].set_xlabel('Time (Month - Day - Hour)', size=14)
    ax[1].legend(['1hr Moving Average', '6hr Moving Average', 'Cumulative Deviations'])

    plt.show()

    test_deviation = test_truth.numpy() - test_preds.numpy()
    test_deviation_upper = test_truth.numpy() - test_upper.numpy()
    test_deviation_lower = test_truth.numpy() - test_lower.numpy()

    test_upper = np.argwhere(test_deviation_upper > 0.0)
    test_lower = np.argwhere(test_deviation_lower < 0.0)

    fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(16, 12))

    ax[0].scatter(test_time, test_deviation, alpha=0.5)
    ax[0].scatter(test_time[test_upper], test_deviation[test_upper], color='red')
    ax[0].scatter(test_time[test_lower], test_deviation[test_lower], color='red')
    ax[0].set_title('Energy Performance Deviations', size=14)
    ax[0].set_ylabel('kW (difference in predicted vs. actual)', size=14)
    ax[0].legend([
        'Actual kW - Predicted kW', 'Upper and Lower Control Limit +/- $2 \sigma$'])

   
    ax[1].scatter(test_time, np.cumsum(test_deviation), alpha=0.5)
    ax[1].plot(test_time, pd.Series(np.cumsum(test_deviation)).rolling(6).mean(), color='orange')
    #ax[1].plot(test_time, pd.Series(np.cumsum(test_deviation)).rolling(60).mean(), color='green')
    ax[1].set_title('Energy Performance Cumulative Deviations', size=14)
    ax[1].set_ylabel('kW (Cumulative difference in predicted vs. actual)', size=14)
    ax[1].set_xlabel('Time (Month - Day - Hour)', size=14)
    ax[1].legend(['1hr Moving Average', 'Cumulative Deviations'])
    
    plt.show()


def main():

    print('call functions above in this function')



if __name__ == "__main__":
    main()