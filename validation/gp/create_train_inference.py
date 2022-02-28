import torch
import gpytorch
from validation.gp.exact_gp import ExactGPModel
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from lib.util import helper
import time
import json
import numpy as np


def create_train_inference_gp(kernel_gen, train_x, train_y, test_x,
    test_y, n_train, machine=str, likelihood_noise=None, training_iter=100, lr=0.1,
    update_score=False, time_agg=None):

    start_time = time.time()

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    if likelihood_noise is not None:
        likelihood.initialize(noise=likelihood_noise)

    model = ExactGPModel(
        train_x,
        train_y,
        likelihood,
        kernel_gen
    )

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

    mse, mape = posterior_inference(train_x, train_y, test_x, test_y, model, \
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

    return model, likelihood, mse, mape


def posterior_inference(train_x, train_y, test_x, test_y, model, likelihood, n_train):

    model.eval()
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        
        observed_preds = likelihood(model(test_x))
        test_preds = observed_preds.mean[n_train:]
        test_x_sub = test_x[n_train:]

        mse = mean_squared_error(test_y.numpy(), test_preds.numpy())
        mape = mean_absolute_percentage_error(test_y.numpy(), test_preds.numpy())

        f, ax = plt.subplots(figsize=(16, 7))
        lower, upper = observed_preds.confidence_region()
        ax.plot(train_x.numpy(), train_y.numpy())
        ax.plot(test_x.numpy(), observed_preds.mean.numpy())
        ax.plot(test_x_sub.numpy(), test_y.numpy(), color='red')
        ax.fill_between(
            test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5, color='#2ecc71'
            )
        plt.axvline(x=np.min(test_x_sub.numpy()), linestyle='--', color='black', lw=1)
        ax.legend([
            'Observed Data', 'Predictions', 'Test Truth', 'Train/Test Split', 'Uncertainty'
            ])
        plt.xlabel('Time (Normalized)')
        plt.ylabel('Standardized kW')
        plt.title(
            'Gaussian Process Regression: Interpolation and Extrapolation'
            )
        plt.show()
    
    return mse, mape