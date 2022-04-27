import torch
import gpytorch
from scripts import plot_results
from scripts import run_experiments


class Experiments:

    def __init__(self, machine, time_agg) -> None:
        
        self.machine = machine
        self.time_agg = time_agg
    

    def plot_experiments():
        plot_results.main()


    def load_model_state(self):
        print('loading model state')
    

    def run_model(self):
        print('running model')

    
    def plot_model(self):
        print('plotting results')


if __name__ == "__main__":

    Experiments()