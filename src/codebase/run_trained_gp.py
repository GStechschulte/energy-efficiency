import os
from re import L
import torch
import gpytorch
import numpy as np
from exact_gp import ExactGPModel
import data_utils
import kernel_utils
from sklearn.metrics import mean_absolute_percentage_error, \
    mean_squared_error, mean_pinball_loss

class Experiments:

    def __init__(self, machine, time_agg) -> None:
        
        self.machine = machine
        self.time_agg = time_agg
        self.training_iter = 100
        self.lr = 0.1
    

    def get_data(self):
        """
        Fetch testing data (6hr windows?) 
        """
        
        X_train, y_train, X_test, y_test, n_train = data_utils.gp_preprocess(
            machine=self.machine, 
            freq=self.time_agg,
            normalize_time=True,
        )

        return X_train, y_train, X_test, y_test, n_train
        
    
    def load_model_state(self):

        cwd = os.getcwd()
        path_model_state = cwd + '/src/saved_models/'

        self.state_dict = torch.load(
            path_model_state + 'entsorgung_10T.pth') ## change to variable

        return self.state_dict
    

    def load_kernel(self):

        self.kernel_function = kernel_utils.entsorgung_kernel()
    
    def load_model(self):
        
        self.X_train, self.y_train, self.X_test, self.y_test, self.n_train = self.get_data()
        self.load_kernel()

        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()

        self.model = ExactGPModel(
            train_x=self.X_train, train_y=self.y_train, 
            likelihood=self.likelihood, kernel=self.kernel_function)

        print('\n', 'Base Model: Before Loading Learned Parameters')
        
        for param_name, param in self.model.named_parameters():
            print(f'Parameter name: {param_name:42} value = {param.item()}')

        self.model.load_state_dict(self.load_model_state())

        print('\n', 'Base Model: After Loading Learned Parameters')

        for param_name, param in self.model.named_parameters():
            print(f'Parameter name: {param_name:42} value = {param.item()}')

        return self

    def inference(self):
        
        self.model.eval()
        self.likelihood.eval()
        
        with torch.no_grad(), gpytorch.settings.fast_pred_var():

            func_preds = self.model(self.X_test)
            observed_preds = self.likelihood(self.model(self.X_test))
            lower, upper = observed_preds.confidence_region()

        # Transform inputs and outputs back to original scales
        train_y_inv, test_y_inv, observed_preds_inv, func_preds_mean_inv, func_preds_var_inv, \
        lower_inv, upper_inv, orig_time, orig_time_train, orig_time_test = \
        data_utils.gp_inverse_transform(
        self.y_train, self.y_test, observed_preds, func_preds, lower, upper)

        self.test_preds = observed_preds_inv[self.n_train:]
        self.lower_preds = lower_inv[self.n_train:]
        self.upper_preds = upper_inv[self.n_train:]

        print(func_preds.mean)

        self.scoring(ground_truth=test_y_inv)


    def scoring(self, ground_truth): ## import this function
        
        mean_pb_loss = mean_pinball_loss(ground_truth, self.test_preds)

        indicator = []
        for x, low, up in zip(ground_truth, self.lower_preds, self.upper_preds):
            if x <= up and x >= low:
                indicator.append(1)
            else:
                indicator.append(0)

        ace = sum(indicator) / len(ground_truth)

        mse = mean_squared_error(ground_truth.numpy(), self.test_preds.numpy())
        mape = mean_absolute_percentage_error(ground_truth.numpy(), self.test_preds.numpy())

        print('\n', 'Evaluation Metrics')
        print('-'*20)
        print('MSE       = ', round(mse, 4))
        print('RMSE      = ', round(np.sqrt(mse)))
        print('MAPE      = ', round(mape, 4))
        print('ACE       = ', round(ace, 4))
        print('Pinball   = ', round(mean_pb_loss, 4))



    def plot_model(self): ## import this function
        print('plotting results')




if __name__ == "__main__":

    exp = Experiments(
        machine='entsorgung_10T',
        time_agg=10
    )

    exp.load_model().inference()
