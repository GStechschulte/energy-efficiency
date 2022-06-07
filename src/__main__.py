from scripts import run_experiment
from scripts import plot_results


class Experiments:


    def __init__(self, machine, time_agg) -> None:
        
        self.machine = machine
        self.time_agg = time_agg
        

    def run_experiments(self):
        
        self.mean, self.lower, self.upper = run_experiment.main(
            machine=self.machine, time=self.time_agg
            )

    def plot_experiments(self):
        plot_results.main(
            machine=self.machine, time=self.time_agg, mean_preds=self.mean,
            lower_preds=self.lower, upper_preds=self.upper
        )


    def run_and_plot(self):
        self.run_experiments()
        self.plot_experiments() 


if __name__ == "__main__":

    machine = str(input('Enter machine name:'))
    time = int(input('Enter time aggregation (10 or 30):'))

    Experiments(machine, time).run_and_plot()