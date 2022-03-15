from re import S
import torch
import gpytorch
import pandas as pd
from validation.gp.exact_gp import ExactGPModel
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
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

    global machine_name
    global time_aggregation
    machine_name = machine
    time_aggregation = time_agg

    start_time = time.time()

    #likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=0.001)
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

    #if time_agg == '5T' or time_agg == '1T':
    #    model.double()
    #    likelihood.double()
    #else:
    #    pass


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
    
    #for param_name, param in model.named_parameters():
    #    print(f'Parameter name: {param_name:42} value = {param.item()}')

    mse, mape, perf_dev_upper \
        = posterior_inference(train_x, train_y, test_x, test_y, model, \
        likelihood, n_train)
    
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
            elapsed_time=elapsed_time
        )
    else:
        pass

    #torch.save(model.state_dict(), '../gpytorch_models/model_state.pth')

    return model, likelihood, mse, mape, perf_dev_upper #, perf_dev_lower


def posterior_inference(train_x, train_y, test_x, test_y, model, likelihood, n_train):
    """
    Get into posterior preditive mode, compute predictions using LOVE, transform 
    inputs and outputs back to their original scale, and plot the results
    
    Parameters
    ----------

    Returns
    -------
    """
    model.eval()
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        
        observed_preds = likelihood(model(test_x))
        lower, upper = observed_preds.confidence_region()

        # Transform inputs and outputs back to original scales
        train_y_inv, test_y_inv, observed_preds_inv, lower_inv, upper_inv, orig_time,  \
            orig_time_train, orig_time_test  = data_preprocessing.gp_inverse_transform(
            train_y, test_y, observed_preds, lower, upper 
        )

        test_preds = observed_preds_inv[n_train:]
        lower_preds = lower_inv[n_train:]
        upper_preds = upper_inv[n_train:]
        
        mse = mean_squared_error(test_y_inv.numpy(), test_preds.numpy())
        mape = mean_absolute_percentage_error(test_y_inv.numpy(), test_preds.numpy())

        f, ax = plt.subplots(figsize=(16, 7))
        training = ax.scatter(orig_time_train, train_y_inv.numpy(), s=[2.75], color='black')
        in_sample = ax.plot(orig_time, observed_preds_inv.numpy())
        out_sample = ax.scatter(orig_time_test, test_y_inv.numpy(), s=[2.75], color='black')

        ci = ax.fill_between(
            orig_time, lower_inv.numpy(), upper_inv.numpy(), alpha=0.5, 
            color='darkgrey'
            )

        split = plt.axvline(x=np.min(orig_time_test), linestyle='--', color='black', lw=1)

        perf_dev_upper, perf_dev_lower = performance_deviation(
        train_y_inv, test_y_inv, orig_time, lower, upper
        )
        
        for time, val in perf_dev_upper.items():
            ax.annotate(
                round(val, 2), (time, val), fontsize=12)

        for time, val in perf_dev_lower.items():
            ax.annotate(
                round(val, 2), (time, val), fontsize=12)

        #ax.legend([
        #    'Observed Data', 'Predictions', 'Test Truth', 'Train/Test Split', 
        #    'Uncertainty'
        #    ])

        mean_total_energy, upper_total_energy, lower_total_energy = \
        forecasted_consumption(
        test_preds,
        lower_preds,
        upper_preds
        )

        plt.xlabel('Time', fontsize=14)
        plt.xticks()
        plt.ylabel('kW', fontsize=14)
        #ax.text(
        #    '2021-10-15', np.max(upper_inv.numpy()) - 0.25,
        #    'Expected next day consumption = {} kWh'.format(np.round(mean_total_energy), 2) 
        #    )
        #plt.ylim(bottom=torch.minimum(y))
        plt.title(
            'Machine: {}, Time Aggregation: {}'.format(
                machine_name, time_aggregation
            )
            )
        plt.show()

        print('{}'.format(machine_name))
        print('-----------------------------------')
        print('Expected next day energy consumption    = ', mean_total_energy, 'kWh')
        print('Upper bound next day energy consumption = ', upper_total_energy, 'kWh')
        print('Lower bound next day energy consumption = ', lower_total_energy, 'kWh')
        print('\n')

        print('Abnormal High Energy Consumption')
        print('---------------------------------')
        for time, val in perf_dev_upper.items():
            print(time, val)

        print('\n') 

        print('Abnormal Low Energy Consumption')
        print('---------------------------------')
        for time, val in perf_dev_lower.items():
            print(time, val) 
    


    
    return mse, mape, perf_dev_upper


def performance_deviation(train_y_inv, test_y_inv, time, lower_inv, upper_inv):
    """
    Find values that lie outside of the confidence region

    Parameters
    ----------

    Returns
    -------
    
    
    """

    orig_time = np.array(time)

    ground_truth = np.concatenate([train_y_inv.numpy(), test_y_inv.numpy()])

    # Upper
    out_of_conf_upper_idx = np.array(
        np.where(ground_truth > upper_inv.numpy())).flatten()

    out_of_conf_upper_vals = ground_truth[out_of_conf_upper_idx]
    out_of_conf_upper_time = orig_time[out_of_conf_upper_idx]

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

    return perf_dev_upper, perf_dev_lower


def forecasted_consumption(preds_mean, lower_inv, upper_inv):
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
    mean_total_energy = torch.sum(preds_mean*0.5).numpy()
    lower_total_energy = torch.sum(lower_inv*0.5).numpy()
    upper_total_energy = torch.sum(upper_inv*0.5).numpy()
    
    # Plot out-of-sample prediction

    return mean_total_energy, upper_total_energy, lower_total_energy





