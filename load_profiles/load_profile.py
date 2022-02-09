%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%%

class load_plotter():

    def __init__(self, df, epoch_convert=bool):
        self.df = df

        if epoch_convert == True:
            self.df = self.utc()
            #self.df = df
        else:
            self.df = df

    def utc(self):

        self.df.t = pd.to_datetime(self.df.t, unit='s')
        self.df.set_index(self.df.t, inplace=True)
        del self.df['t']
        
        return self.df

    def plot(self, feat, phase, resample=None):
        
        if resample != None:
            plt.figure(figsize=(16, 6))
            plt.plot(self.df.resample(resample).sum().index, self.df[feat].resample(resample).mean())
            plt.title('Phase {}'.format(str(phase)))
            plt.xlabel('Time = M-d-H')
            plt.ylabel('Power')
            plt.show()

        else:
            plt.figure(figsize=(16, 6))
            plt.plot(self.df.index.iloc[self.df.L == phase], self.df[feat].iloc[self.df.L == phase])
            plt.title('Phase {}'.format(str(phase)))
            plt.xlabel('Time = M-d-H')
            plt.ylabel('{}'.format(feat))
            plt.show()            