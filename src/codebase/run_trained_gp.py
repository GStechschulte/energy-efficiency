import os
import torch
import gpytorch
from exact_gp import ExactGPModel
import data_utils
import kernel_utils

class Experiments:

    def __init__(self, machine, time_agg) -> None:
        
        self.machine = machine
        self.time_agg = time_agg
        self.training_iter = 100
        self.lr = 0.1
    

    def get_data(self):
        
        X_train, y_train, X_test, y_test, n_train = data_utils.gp_preprocess(
            machine=self.machine, 
            freq=self.time_agg,
            normalize_time=True,
        )

        return X_train, y_train, X_test, y_test
        
    
    def load_model_state(self):

        cwd = os.getcwd()
        path_model_state = cwd + '/src/saved_models/'

        self.state_dict = torch.load(
            path_model_state + 'Entsorgung_10T.pth')

        return self.state_dict
    

    def load_kernel(self):

        self.kernel_function = kernel_utils.entsorgung_kernel()

        return self.kernel_function
    
    def load_model(self):
        
        self.X_train, self.y_train, self.X_test,self.y_test = self.get_data()
        kernel_function = self.load_kernel()

        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()

        self.model = ExactGPModel(
            train_x=self.X_train, train_y=self.y_train, 
            likelihood=self.likelihood, kernel=self.kernel_function)

        self.model.load_state_dict(self.load_model_state())

        return self


    def perform_training(self):
        
        self.model.train()
        self.likelihood.train()

        # Loss function for GPs
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        # Optimization method --> Adam
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        for i in range(self.training_iter):
            optimizer.zero_grad()
            output = self.model(self.X_train)
            loss = -mll(output, self.y_train)
            loss.backward()

            print('Iter {} , Loss = {} , Noise = {}'.format(
            i+1, loss, self.model.likelihood.noise.item() 
            ))

            optimizer.step()


    
    def plot_model(self):
        print('plotting results')


if __name__ == "__main__":

    exp = Experiments(
        machine='entsorgung_10T',
        time_agg=10
    )

    exp.load_model().perform_training()
