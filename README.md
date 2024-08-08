# PyQUANT3
Python version of QUANT3 for DAFNI SCQUAIR Project

# INSTALLATION
Run the following to install the matrices in the model-runs dir from the store on the OSF.IO site:

cd scripts
python install.py

This will check whether existing files are present and only download if they are missing.
This is the automated version of the desctiption in the next section "File Inputs (Manual version)" which explains what the
script does at a lower level.

# File Inputs (Manual version)
The matrices need to be downloaded from here: https://osf.io/x2gbn/ (if not already installed using install.py)
These are put in a directory called model-runs
On the DAFNI system, this will be /data/inputs/model-runs with the matrices supplied by the model runner system on DAFNI. When running locally, the directory is hard coded in pyquant3.py (todo: this needs to be changed).

appsettings.yaml contains a ModelRunsDir parameter that contains the name of the directory under the inputs directory that contains the matrix data below. This is probably going to be called model-runs unless you have a very good reason to change it.

TObs_1.bin Matrix flow file (road mode) containing trips between all pairs of zones (8436x8436)
TObs_2.bin Matrix flow file (bus mode) containing trips between all pairs of zones (8436x8436)
TObs_3.bin Matrix flow file (rail mode) containing trips between all pairs of zones (8436x8436)

dis_roads_min.bin Costs file (road mode) containing time in minutes to transit between all pairs of zones (8436x8346)
dis_bus_min.bin Costs file (bus mode) containing time in minutes to transit between all pairs of zones (8436x8346)
dis_gbrail_min.bin Costs file (rail mode) containing time in minutes to transit between all pairs of zones (8436x8346)

dis_crowfly_vertex_roads_KM.bin Distance (road mode) in kilometres between the network vertices used for routing between zones (8436x8436)
dis_crowfly_vertex_bus_KM.bin Distance (bus mode) in kilometres between the network vertices used for routing between zones (8436x8436)
dis_crowfly_vertex_gbrail_KM.bin Distance (rail mode) in kilometres between the network vertices used for routing between zones (8436x8436)

EWS_ZoneCodes.xml (.xsd) Lookup between zone MSOA/IZ code and zone numbers needed for the matrix files (above).

# Command Line Options
-h, --help prints help text
-d, --dafni if this option is present, then assume we're running on DAFNI and set the IsOnDAFNI environment option, otherwise we're running locally in debug mode
-o, --opcode pass in --opcode CALIBRATE | RUN to perform a calibration or a scenario run
--betaroad beta value for road mode from outputs/calibration.yaml e.g. --betaroad 0.1316692928026544 
--betabus beta value for bus mode from outputs/calibration.yaml e.g. --betabus 0.0728867427898217
--betarail beta value for rail mode from outputs/calibration.yaml e.g. --betarail 0.06495053139819612

# Outputs
In the /data/outputs directory for DAFNI, or wherever defined for a local run:

calibration.yaml beta values and means trip lengths from a model calibration
debug.txt debugging information - not used except during testing
PyQUANT3_log.txt detailed logging information, including exception traces
impacts_yyyymmdd_hhmmss.csv time-stamped scenario impacts data from a batch of scenario runs
