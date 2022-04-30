from functools import update_wrapper
from re import S
import os
from statistics import mode
from matplotlib.cbook import silent_list
from pyparsing import col
import torch
import gpytorch
import pandas as pd
from tqdm import tqdm
from validation.gp.exact_gp import ExactGPModel
from validation.gp.trace_model import MeanVarModelWrapper
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_pinball_loss
from scipy import linalg
from lib.util import helper
from lib.util import data_preprocessing
import time
import json
import numpy as np


machine_name = None
time_aggregation = None


def create_train_inference_gp(kernel_gen, train_x, train_y, test_x,
    test_y, n_train, machine=str, likelihood_noise=None, training_iter=100, lr=0.1,
    update_score=False, time_agg=None):
    """
    
    Parameters
    ----------

    Returns
    -------

    """

    global machine_name
    global time_aggregation
    machine_name = machine
    time_aggregation = time_agg

    start_time = time.time()
    
    #likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=0.005)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()

    if likelihood_noise is not None:
        likelihood.initialize(noise=likelihood_noise)

    model = ExactGPModel(
        train_x,
        train_y,
        likelihood,
        kernel_gen
    )
    
    #for param_name, param in model.named_parameters():
    #    print(f'Parameter name: {param_name:42} value = {param.item()}')
    """
    if time_agg == '5T' or time_agg == '1T':
        model.double()
        likelihood.double()
        model.train()
        likelihood.train()

        mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.size(0))

    else:
        model.train()
        likelihood.train()

        # Loss function for GPs
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        # Optimization method --> Adam
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for i in range(training_iter):
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y)
            loss.backward()

            print('Iter {} , Loss = {} , Noise = {}'.format(
            i+1, loss, model.likelihood.noise.item() 
            ))

            optimizer.step()
    """

    
    model_state = machine_name + '_' + time_aggregation + '.pth'
    
    #torch.save(
    #    model.state_dict(), 
    #    os.path.join(os.path.dirname(__file__), '../gpytorch_models/{}'.format(
    #        model_state)))
    
    state_dict = torch.load(
        os.path.join(os.path.dirname(__file__), '../gpytorch_models/{}'.format(
            model_state)))

    #likelihood = gpytorch.likelihoods.GaussianLikelihood()
    #model = ExactGPModel(
    #    train_x,
    #    train_y,
    #    likelihood,
    #    kernel_gen)

    print(state_dict)
    model.load_state_dict(state_dict)
    
    #print(model.state_dict())

    #traced_model.save(
    #    os.path.join(os.path.dirname(__file__), '../gpytorch_models/{}'.format(
    #    model_trace
    #)))
    #traced_model.save('traced_exact_gp.pt')
    #model.train()
    

    model.eval()
    likelihood.eval()

    func_preds_mean_inv, func_preds_var_inv, observed_preds, mse, mape, out_sample_ace, \
        out_sample_pinball = posterior_inference(train_x, train_y, test_x, test_y, model, likelihood, n_train)
    
    elapsed_time = time.time() - start_time
    
    if update_score is True:
        model_name = 'gp_' + machine

        model_state = {}
        params = model.state_dict()
        for key, value in params.items():
            model_state[key] = value.tolist()

        helper.update_gp_metrics(
            model=model_name,
            time_agg=time_agg,
            kernel=json.dumps(model_state),
            lr=lr,
            training_iter=training_iter,
            optimizer='Adam',
            out_sample_mse=mse,
            out_sample_mape=mape,
            out_sample_ace=out_sample_ace,
            out_sample_pinball=out_sample_pinball,
            elapsed_time=elapsed_time
        )
    else:
        pass

    return func_preds_mean_inv, func_preds_var_inv, observed_preds, mse, mape


def posterior_inference(train_x, train_y, test_x, test_y, model, likelihood, n_train):
    """
    Get into posterior predictive mode, compute predictions using LOVE, transform 
    inputs and outputs back to their original scale, and plot the results
    
    Parameters
    ----------

    Returns
    -------
    """

    with torch.no_grad(), gpytorch.settings.fast_pred_var():

        model.eval()
        likelihood.eval()

        func_preds = model(test_x)
        #traced_model = torch.jit.trace(MeanVarModelWrapper(model), test_x)
        observed_preds = likelihood(model(test_x))
        lower, upper = observed_preds.confidence_region()

        # Transform inputs and outputs back to original scales
        train_y_inv, test_y_inv, observed_preds_inv, func_preds_mean_inv, func_preds_var_inv, \
            lower_inv, upper_inv, orig_time, orig_time_train, orig_time_test = \
            data_preprocessing.gp_inverse_transform(
            train_y, test_y, observed_preds, func_preds, lower, upper
        )

        test_preds = observed_preds_inv[n_train:]
        lower_preds = lower_inv[n_train:]
        upper_preds = upper_inv[n_train:]

        mse = mean_squared_error(test_y_inv.numpy(), test_preds.numpy())
        mape = mean_absolute_percentage_error(test_y_inv.numpy(), test_preds.numpy())
        rmse = np.sqrt(mse)
        cv_rmse = np.mean(test_y_inv.numpy()) * rmse

        out_sample_ace, out_sample_pinball = probability_metrics(
            preds_mean=test_preds.numpy(), lower=lower_preds.numpy(), upper=upper_preds.numpy(), 
            ground_truth=test_y_inv.numpy())

        interpolation_extrapolation(
            orig_time_train, train_y_inv, observed_preds_inv, n_train, upper_inv,
            lower_inv, test_y_inv, orig_time_test, orig_time)

        in_sample_fit(
            orig_time_train, train_y_inv, observed_preds_inv, n_train, upper_inv,
            lower_inv, test_y_inv)

        quality_control(
            preds_mean=observed_preds_inv[:n_train], lower=lower_inv[:n_train],
            upper=upper_inv[:n_train], ground_truth=train_y_inv, 
            orig_time=orig_time_train, test_preds=test_preds, test_lower=lower_preds,
            test_upper=upper_preds, test_truth=test_y_inv, test_time=orig_time_test)

        forecasted_consumption(
            preds_mean=test_preds, test_time=orig_time_test, 
            lower_inv=lower_inv[n_train:], upper_inv=upper_inv[n_train:])
    
    #model_state = machine_name + '_' + time_aggregation + '.pth'
    
    #torch.save(
    #    model.state_dict(), 
    #    os.path.join(os.path.dirname(__file__), '../gpytorch_models/{}'.format(
    #        model_state)))
    

    #print(state_dict)
    #model.load_state_dict(state_dict)

    print(out_sample_ace, out_sample_pinball)

    return func_preds_mean_inv, func_preds_var_inv, observed_preds, mse, mape, out_sample_ace, out_sample_pinball


def interpolation_extrapolation(orig_time_train, train_y_inv, observed_preds_inv, 
n_train, upper_inv, lower_inv, test_y_inv, orig_time_test, orig_time):

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


def in_sample_fit(orig_time_train, train_y_inv, observed_preds_inv, n_train,
upper_inv, lower_inv, test_y_inv):

        f, ax = plt.subplots(figsize=(16, 7))
        training = ax.scatter(orig_time_train, train_y_inv.numpy(), s=[2.75], color='black')
        in_sample = ax.plot(
            orig_time_train, observed_preds_inv[:n_train].numpy(),
            marker='.')

        ci = ax.fill_between(
            orig_time_train, lower_inv[:n_train].numpy(), upper_inv[:n_train].numpy(), alpha=0.5, 
            color='darkgrey'
            )

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

        plt.xlabel('Time', fontsize=14)
        plt.xticks()
        plt.ylabel('kW', fontsize=14)
        plt.title(
            'Machine: {} \n Time Aggregation: {}'.format(
                machine_name, time_aggregation))
        plt.show()


def performance_deviation_training(train_y_inv, observed_preds_inv, time, lower_inv, upper_inv):
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

    orig_time = np.array(time)

    #ground_truth = np.concatenate([train_y_inv.numpy(), test_y_inv.numpy()])
    ground_truth = train_y_inv.numpy()

    # Upper
    out_of_conf_upper_idx = np.array(
        np.where(ground_truth > upper_inv.numpy())).flatten()

    out_of_conf_upper_vals = ground_truth[out_of_conf_upper_idx]
    out_of_conf_upper_time = orig_time[out_of_conf_upper_idx]
    out_of_conf_upper_preds = observed_preds_inv[out_of_conf_upper_idx]
    

    perf_dev_upper = {}
    for val, time in zip(out_of_conf_upper_vals, out_of_conf_upper_time):
        perf_dev_upper[time] = val

    # Lower
    out_of_conf_lower_idx = np.array(
        np.where(ground_truth < lower_inv.numpy())).flatten()
    
    out_of_conf_lower_vals = ground_truth[out_of_conf_lower_idx]
    out_of_conf_lower_time = orig_time[out_of_conf_lower_idx]
 
    perf_dev_lower = {}
    for val, time in zip(out_of_conf_lower_vals, out_of_conf_lower_time):
        perf_dev_lower[time] = val

    # Training Registry
    pdd_registry = pd.DataFrame(list(perf_dev_upper.items()), columns=['time', 'actual'])
    pdd_registry['machine'] = machine_name
    print(pdd_registry)

    # Tesing Registry



    return perf_dev_upper, perf_dev_lower


def performance_deviation_testing(test_y_inv, test_preds, test_time, test_lower, test_upper):
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


def forecasted_consumption(preds_mean, test_time, lower_inv, upper_inv):
    """
    Communicates the predicted electricity consumption for the next
    day with uncertainties

    Parameters
    ----------

    Returns
    -------

    """

    # Time of peak load

    # Expected peak load value

    # Duration of high energy consumption

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

    return mean_total_energy, upper_total_energy, lower_total_energy


def quality_control(preds_mean, upper, lower, ground_truth, orig_time,
test_preds, test_upper, test_lower, test_truth, test_time):
    """
    ## Within CI (2 stddev) ##

    If truth < upper, then truth - upper = negative value
    If truth > lower, then truth - lower = positive value

    ## Out of CI (2 stddev) ##

    If truth > upper, then truth - upper = positive value
    If truth < lower, then truth - lower = negative value
    """

    performance_deviation_testing(
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

  
def probability_metrics(preds_mean, upper, lower, ground_truth):
    """
    """

    mean_pb_loss = mean_pinball_loss(ground_truth, preds_mean)

    indicator = []
    for x, low, up in zip(ground_truth, lower, upper):
        if x <= up and x >= low:
            indicator.append(1)
        else:
            indicator.append(0)

    ace = sum(indicator) / len(ground_truth)

    return ace, mean_pb_loss