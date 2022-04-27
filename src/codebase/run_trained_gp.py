import os
#from re import L
import torch
import gpytorch
import numpy as np
from exact_gp import ExactGPModel
import data_utils
import kernel_utils
from scoring import scoring_metrics
#import plot_results


class Experiments:

    def __init__(self, machine, time_agg) -> None:
        
        self.machine = machine
        self.machine_name = machine + '_' + str(time) + 'T'
        self.time_agg = time_agg
        self.training_iter = 100
        self.lr = 0.1
    

    def get_data(self):
        """
        
        """
        
        self.X_train, self.y_train, self.X_test, self.y_test, self.n_train = data_utils.gp_preprocess(
            machine=self.machine_name, 
            freq=self.time_agg,
            normalize_time=True)
        
    
    def load_model_state(self):

        cwd = os.getcwd()
        path_model_state = cwd + '/src/saved_models/'

        self.state_dict = torch.load(
            path_model_state + self.machine_name +'.pth')

        return self.state_dict
    

    def load_kernel(self):

        self.kernel_function = kernel_utils.main(
            machine=self.machine, time=self.time_agg
            )
    
    
    def load_model(self):
        
        self.get_data()
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

        #print(func_preds.mean)

        self.scoring(ground_truth=test_y_inv)


    def scoring(self, ground_truth):
        
        scoring_metrics(
            ground_truth=ground_truth,
            test_preds=self.test_preds,
            upper_preds=self.upper_preds,
            lower_preds=self.lower_preds)
        

    #def plot_experiments(self): 
    #    plot_results.main()




if __name__ == "__main__":

    machine = str(input('Enter machine name:'))
    time = int(input('Enter time aggregation (10 or 30):'))

    exp = Experiments(
        machine=machine,
        time_agg=time
    )

    exp.load_model().inference()
