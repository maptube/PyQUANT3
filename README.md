# PyQUANT3
Python version of QUANT3 for DAFNI SCQUAIR Project

# Inputs
The matrices need to be downloaded from here: https://osf.io/x2gbn/
These are put in a directory called model-runs
On the DAFNI system, this will be /data/inputs/model-runs with the matrices supplied by the model runner system on DAFNI. When running locally, the directory is hard coded in pyquant3.py (todo: this needs to be changed).

appsettings.yaml contains a ModelRunsDir parameter that contains the name of the directory under the inputs directory that contains the matrix data below. This is probably going to be called model-runs unless you have a very good reason to change it.

TObs_1.bin Matrix flow file (road mode) containing trips between all pairs of zones (8436x8436)
TObs_2.bin Matrix flow file (bus mode) containing trips between all pairs of zones (8436x8436)
TObs_3.bin Matrix flow file (rail mode) containing trips between all pairs of zones (8436x8436)

dis_roads_min.bin Costs file (road mode) containing time in minutes to transit between all pairs of zones (8436x8346)
dis_bus_min.bin Costs file (bus mode) containing time in minutes to transit between all pairs of zones (8436x8346)
dis_gbrail_min.bin Costs file (rail mode) containing time in minutes to transit between all pairs of zones (8436x8346)

EWS_ZoneCodes.csv Lookup between zone MSOA/IZ code and zone numbers needed for the matrix files (above).
