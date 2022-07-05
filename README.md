# Estimation of energy efficiency and condition monitoring #

We conducted a preliminary study, *"Condition monitoring and energy efficiency tools from smart (sub-)meters"*, to gauge the potential of condition monitoring and energy efficiency digital services based mainly on electricity consumption data. Thus, we tested methods that could be used as the basis to build estimation tools with a particular focus on industrial sites and factory equipment (industry 4.0). 

## Background and objectives ##

## About this repository ##
Code, works in progress, and supplemental information related to Gabriel Stechschulte's master thesis *"Industrial Equipment Energy Efficiency Estimation and Performance Deviation Detection Using IoT Enabled Energy Metering"* and research performed in the context of the innovation project. 

### Directories ###

[clemap_api](clemap_api/) - Notebook for authenticating a new user

[data](data/) - Data folder with pointers or simply descriptions of data sets

[data_loader](data_loader/) - Python script for loading all the machine data into PostgreSQL

[docs_in_switch](docs_in_switch/) - Administrative related files stored in shared drive

[EDA](EDA/) - Contains visualizations of load profiles broken down by machine, day, and hour

[examples](examples/) - [to be deprecated?]

[figs](figs/) - Plots saved during experimentation

[gam_validation](gam_validation/) - Hyperparameter tunning of GAM models (fbprophet)

[gp_validation](gp_validation/) - GP notebooks for modelling `entsorgung`, `uv_eg`, `group_1` and `group_2`

[lib](lib/) - Utility or helper functions

[load_profiles](load_profiles/) - Scripts to visualize 24 hr-snippets of time series

[metad](metad/) - Data folder with metadata of datasets in `data` folder and CLEMAP's measurement setup 

[models](models/) - Clustering and time series models of various machines, as well as examples for sampling from priors

[report](report/) - Gabriel's master thesis

[sql](sql/) - ...

[src](src/) - ...

[validation](validation/) - ...

### Branches ###

* `main`: contains a tutorial to reproduce the prediction results based on a docker container 

* `experiments`: contains also work done in the context of the innosuisse project, it is the main "working" branch 

* `models` contains GAM and GP validation notebooks

* `load_profiles`, `temporal` are old versions of `experiments`

