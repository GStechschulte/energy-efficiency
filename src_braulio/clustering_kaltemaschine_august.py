import os
from matplotlib import pyplot as plt
import numpy as np
import glob
import pandas as pd
import time
"""
from pycaret.utils import version
version()
"""
from pycaret.clustering import *

# my path: /Users/tabaraho/clemap/Energy-Efficiency-Thesis/src/overview.py
home = "/Users/tabaraho/clemap/Energy-Efficiency-Thesis/src"
os.chdir(home)

# load data file provided by Gino
ddir = "../data/20210808-v0.0.1_SE05000139_Kaeltemaschine" # data directory
 
# print(glob.glob(ddir + os.sep + "*.txt"))
fns = glob.glob(ddir + os.sep + "*.csv")

D = []
for f in np.sort(fns):
  
  df = pd.read_csv(f)
  # these time stamps are in seconds from the epoch, e.g. UTC
  df["ts"] = pd.to_datetime(df.t, unit="s")
  
  #print(f)
  #print(pd.read_csv(f).info())
  #print(list(df))

  df = df.pivot(index="ts",columns="L", values="P").sort_index()
  D.append(df)

D = pd.concat(D) # power per-phase
D["P"]= D.sum(axis=1)
P_1s   =D.resample("1s", label="right").mean()
P_1min =D.resample("1min", label="right").mean()
P_15min=D.resample("15min", label="right").mean()
P_1hr  =D.resample("1H", label="right").mean()

# illustration of unsupervised learning to cluster machine operating modes
power = P_1min*1e-3 # Select from different resolutions created above
N_cl = 3 # Number of clustes, I tried 3 and 4 with and without normalization
flg_norm = False

if not type(power) == pd.core.frame.DataFrame: # *!* something is buggy and I fix it like this :)
  data = pd.DataFrame(power) # power data single feature, 3 days
else:
  data = power

# include day as a feature, when using N_cl = 4 to reproduce examples shown in slides
#data["day"] = P_1min.index.day_name().values  
clu1 = setup(data, session_id=123, log_experiment=True, log_plots = True, 
             experiment_name='test_1',
             normalize=flg_norm, ignore_features=["P"])


models() # Create model
kmeans = create_model('kmeans', num_clusters = N_cl)

kmeans_results = assign_model(kmeans) # Assign Labels
kmeans_results.head()

#kmodes = create_model('kmodes', num_clusters = N_cl)
#kmodes_results = assign_model(kmodes)
#kmodes_results.head()

p_cl = kmeans_results.pivot(columns="Cluster", values="P") 
plt.figure()
plt.plot(power.P, "-k", linewidth=0.2)
for ci in list(p_cl):
  plt.plot(p_cl[ci], label=ci)
plt.legend()
plt.ylabel("P [kW]")
plt.show()

### (*!) plt.savefig("../figs/1_SE05000139_Kaeltemaschine_P1min_Ncl4_NormalizationTrue_zoom.png", dpi=600)
# Notes:
#   3 full days from Saturday 7th in the evening to Tuesday 10th of August
