import os
import torch
import gpytorch
from .exact_gp import ExactGPModel
from .scoring import scoring_metrics
from . import data_utils
from . import kernel_utils


def perform_inference(model, likelihood, x_test):

    model.eval()
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        func_preds = model(x_test)
        observed_preds = likelihood(model(x_test))
        lower, upper = observed_preds.confidence_region()

    return observed_preds.mean, lower, upper


def main(machine=str, time=int):
    
    machine_name = machine + '_' + str(time) + 'T'
    print('Building {} GP model'.format(machine_name))

    # Directory path to saved model parameters
    cwd = os.getcwd()
    path_model_state = cwd + '/src/saved_models/'

    # Load model parameters
    state_dict = torch.load(
            path_model_state + machine_name +'.pth'
            )

    # Get kernel function
    kernel_function = kernel_utils.main(
            machine=machine, time=time
            )

    # Query data + preprocess
    X_train, y_train, X_test, y_test, n_train = \
        data_utils.gp_preprocess(
            machine=machine_name, 
            freq=time,
            normalize_time=True)

    likelihood = gpytorch.likelihoods.GaussianLikelihood()

    # Initialize GP model
    model = ExactGPModel(
        train_x=X_train, train_y=y_train, 
        likelihood=likelihood, kernel=kernel_function
        )

    # Load trained parameters into model
    model.load_state_dict(state_dict)
    print('{} GP model is ready'.format(machine_name))

    # Perform inference
    mean, lower, upper = perform_inference(model, likelihood, X_test)

    # Evaluation metrics
    scoring_metrics(
        y_test, mean[n_train:], lower[n_train:], upper[n_train:]
        )

    return mean, lower, upper


if __name__ == '__main__':
    main()