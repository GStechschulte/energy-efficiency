""" ipython script for parsing *.tar.gz file 
  Use gzip and tar
  # uncompress
  gzip -d gassmann_data_from_20211011_to_20211017.tar.gz 

  # list and extract
  tar -tvf gassmann_data_from_20211011_to_20211017.tar 
  tar -xvf gassmann_data_from_20211011_to_20211017.tar
"""
from dataclasses import replace
import glob
import os
datad = "/run/user/1005/gvfs/smb-share:server=fs01e.eee.intern,share=data$/22 TES/Datenaustausch_clemap/data"
#fs = os.listdir(datad)
fs = glob.glob(os.path.join(datad, "*.csv.gz"))

for i in fs:
  !gzip -d {i.replace(" ", "\ ")} # removes *.zip files!

fs=glob.glob(os.path.join(datad, "*.csv"))
print(len(fs))

# organizing the directories #
!mkdir {"/run/user/1005/gvfs/smb-share:server=fs01e.eee.intern,share=data$/22 TES/Datenaustausch_clemap/data/20211011_to_20211017".replace(" ", "\ ")}

_new_path = os.path.join(datad, "20211011_to_20211017")

for fi in fs:
  file_name = os.path.basename(fi)
  _fpath = os.path.join(_new_path, file_name)
  #!mv -f {fi.replace(" ", "\ ")} {_fpath.replace(" ", "\ ")}
  os.rename(fi, _fpath)
print(_fpath)
