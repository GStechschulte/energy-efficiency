import os
from matplotlib import pyplot as plt
import numpy as np
import glob
import pandas as pd
import time
from collections import defaultdict

def read_machine_data(fns):
  D = []
  for f in np.sort(fns):

    print(i)

    df = pd.read_csv(f)
    # these time stamps are in seconds from the epoch, e.g. UTC
    df["ts"] = pd.to_datetime(df.t, unit="s")
    df = df.pivot(index="ts",columns="L", values="P").sort_index()
    D.append(df)

  return pd.concat(D) # power per-phase

"""
class MyPath():
  def __setattr__(self, __name: str, __value: Any) -> None:
      pass
  def __init__(self) -> None:
      pass
"""

home = "/home/tabaraho/energy-efficiency/src"
homeX = "/run/user/1005/gvfs/smb-share:server=fs01e.eee.intern,share=data$/22 TES/Datenaustausch_clemap"
os.chdir(home)
# /run/user/1005/gvfs/smb-share:server=fs01e.eee.intern,share=data$/22\ TES/Datenaustausch_clemap
# Load data files and metadata
ddir = os.path.join(homeX, "data/20211125") # directory for storage large files
print(os.path.isdir(ddir))

mdir = os.path.join(homeX, "metad/20211125") # metadata directory 
                           # *!* not in git to avoid exposing sensitive info
md_fn0 = "ErfassungEnMessung_Installed_extended+AF+BB_v2.xlsx"

""" Sensor list from text file
  md_fn1 = "sensor_list.txt"
  sensors = pd.read_csv(os.path.join(mdir, md_fn1), header=None, names=["CLEMAP DB ID"])
  # *!* missing sensor: 5fe3407e923d596335e69d45;SE05000132;Old sensor, replaced by SE05000152
  #MD=list(pd.read_excel(os.path.join(mdir, md_fn0), sheet_name=None))
  map_ids = pd.read_excel(os.path.join(mdir, md_fn0), 
    sheet_name='CLAP_Join_sensorid&hw_serial') 
"""
f_md = os.path.join(mdir, md_fn0)
map_all = pd.read_excel(f_md, converters={"CLEMAP DB ID": str, 
  "Name": str, "Subsystem": str, "Client ID - Serial N.": str}, sheet_name='original_extended') 

map_all[["CLEMAP DB ID", "Name", "Subsystem", "Client ID - Serial N.", "Etage", "Location"]]

#_msub = map_all["Subsystem"].str.split(",")
#map_all["main_sub"] = [i[0] for i in _msub]

fns = glob.glob(ddir + os.sep + "*.csv")

# For every name save the file names 
Dfns = defaultdict(list)
valid_fn=[]
nfs=0
for m, id in zip(map_all["Name"], map_all["CLEMAP DB ID"]):
  for f in fns:
    if str(id) in f:
      Dfns[m].append(f)
      nfs+=1
      valid_fn.append(f)

DFs = {}
for i in Dfns.items():
  print("Reading data from {}".format(i[0]))
  DFs[i[0]] = read_machine_data(i[1]) 

# Calculate total power and resample data
DFs_100ms = {}
Psum = {}
for machine, data in DFs.items():
  # data: raw values per-phase, assume they are in W
  # P_100ms: data resampled to 0.1s and converted to kW
  # 
  P_100ms = (data.sum(axis=1)*1e-3).resample("100ms", label="right").mean()
  Psum[machine] = np.sum(P_100ms*1e-6)
  DFs_100ms[machine] = P_100ms

  print(machine)
  print(np.sum(P_100ms*1e-6))
  #P_100ms.describe()

df = pd.concat(DFs_100ms).reset_index()

# hampel
# WH = 20 # Hampel window, 20 points at 100ms resolution is a 2 s window
# thr_q = .998 #  quantile for threshold selection
# Wm = "10s" # "meta" window for recursive MAD

""" MAD and Hampel filter 
sinputs
------
WH : window size
t : threshold

_median_WH = P_100ms.rolling(WH).median()
MAD = (np.abs( P_100ms - _median_WH)).rolling(WH).median()
Hampel_indicator = np.heaviside(MAD - t, 1)
"""
# MAD = (np.abs(P_100ms - P_100ms.rolling(WH).median())).rolling(WH).median()
# flg0 = MAD/np.nanmax(MAD)
# flg1R = flg0.resample(Wm, label="right").mean()
# flg1L = flg0.resample(Wm, label="left").mean()

""" plots
  plt.plot(flg0)
  plt.plot(flg1L)
  plt.plot(flg1R)
  plt.plot(P_100ms/np.nanmax(P_100ms))
"""

# Select threshold on normalized MAD
# hi = np.nanquantile(np.sort(flg0), thr_q) 
# out0 = np.heaviside(flg0 - hi,1)

# plt.close("all")
# plt.figure(figsize=(6,4))
# plt.plot(P_100ms, linewidth=0.5, label="100 ms data")
# plt.plot(out0*P_100ms, "o-", markersize=1, linewidth=3, alpha=0.3,
#   label="outlier indicator")
# plt.legend()
# plt.ylabel("P [kW]")
# plt.title(machine)
# plt.tick_params(axis="x", rotation=90)
# plt.tight_layout()

#plt.savefig("../figs/"+machine+"_hampel1.png", dpi=600)
#plt.savefig("../figs/"+machine+"_hampel_zoom1.png", dpi=600)
#plt.savefig("../figs/"+machine+"_hampel_zoom2.png", dpi=600)


""" Recursive MAD """
# _outWm = out0.resample(Wm).sum()
# hi = np.nanquantile(_outWm, thr_q) 
# outWm = np.heaviside(_outWm - hi,1)

# plt.close("all")
# plt.figure(figsize=(6,4))
# _power = P_100ms.resample(Wm).mean()
# plt.plot(_power, linewidth=0.5, label=Wm+" data")
# #plt.plot((outWm*P_100ms).dropna(), "o-", markersize=4, linewidth=4, alpha=0.3,
# #  label="outlier indicator")
# plt.plot((outWm*_power).dropna(), "x-", markersize=4, linewidth=4, alpha=0.3,
#   label="cascade outlier indicator")
# plt.legend()
# plt.ylabel("P [kW]")
# plt.title(machine)
# plt.tick_params(axis="x", rotation=90)
# plt.tight_layout()
# #plt.savefig("../figs/"+machine+"_cascade_hampel1.png", dpi=600)
# #plt.savefig("../figs/"+machine+"_cascade_hampel1_zoom.png", dpi=600)

