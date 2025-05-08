# PyQUANT3
Python version of QUANT3 for the DAFNI SCQUAIR Project.

QUANT is a spatial interaction, or gravity mode, of travel to work in England, Scotland and Wales. The raw data is based on Middle Layer Super Output Area (MSOA), along with the travel to work flows from the 2011 Census, which contains around 20 million workers. In addition to this, travel costs between zones are computed from network shortest path time on the road network, bus network and the rail network. Road times come from the road speed limits and time taken to drive the shortest path in terms of time. The bus and rail networks are built from the published timetables with time between nodes taken as the weight of the network when computing shortest paths.

Calibration fits three beta parameters to the flow and network cost data in order to fit the average travel time for each mode. There is one beta for road, bus and rail. These do no change, so can be passed directly into the QUANT model run to save having to calibrate for every scenario that gets run.

The reason for running QUANT is to build a scenario based on changing the number of jobs in a zone, or the travel time between zones on any of the modes. For example, QUANT can be used to run a Heathrow 3rd runway scenario to predict where people will want to live in order to commute to the new jobs offered by the airport, or it can also be used to run a Crossrail or High Speed 2 scenario, where travel times between zones are changed due to new infrastructure.

In addition to calibration, there are two basic ways of running QUANT: run batches of computer generated scenarios with aggregate impacts, or run a single scenario with impacts generated for every zone which can then be plotted on a map.

QUANT is built at the level of MSOA zones for England and Wales, and Intermediate Zones (IZ) for Scotland. This results in 8436 zones in the model. Each matrix for flows or shortest path costs is 8435x8345 elements.


# INSTALLATION
Run the following to install the matrices in the model-runs dir from the store on the OSF.IO site:

cd scripts
python install.py

This will check whether existing files are present and only download if they are missing.
This is the automated version of the desctiption in the next section "File Inputs (Manual version)" which explains what the script does at a lower level.

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

# DAFNI Environment Variables
BetaRoad (default=0.0) - Beta value for road, if 0.0 then triggers calibration
BetaBus (default=0.0) - Beta value for bus, if 0.0 then triggers calibration
BetaRail (default=0.0) - Beta value for rail, if 0.0 then triggers calibration

(NOTE: all SG_ variables refer to the Scenario Generator under the RUN OpCode)
SG_NumIterations (default 10) - Number of iterations to run for this batch of model runs - can be as high as 50,000 if you want
SG_Mode (default 0) - Transport mode where 0=Road, 1=Bus and 2=Rail
SG_RadiusKM (default 5.0) - Maximum radius of link when running batches of computer generated scenarios - 5KM is a good choice
SG_SpeedKPH (default 100.0) - Speed of new links being created for computer generated scenarios in Kilometres per hour
SG_Start_i (default 0) - range 0..8435, this is the origin zone number to start from when running sequential batches of computer generated scenarios
SG_Start_j (default -1) - range -1..8435, this is the destination zone number to start from when running sequential batches of scenarios. The -1 is a quirk of the scenario generator, where it pre-increments, so passing in -1 means it actually starts at 0. This allows you to take the finish i and j off a previous batch and pass the same i and j to the next batch to continue where the previous batch finished.
SG_NumLinks (default 1) - the number of connected links in computer generated scenarios. NOTE: for 1 link scenarios QUANT uses sequential i and j origin and destination zone numbers to do a full sweep of all possible scenarios. When NumLinks>1, a full sweep is not practical due to the high number of possible scenarios, so this runs a different scenario generator which produces random i and j values. This overrides start_i and start_j when numlinks>1.
SG_Network (default '') - Runs a one off scenario from the graphml file specified by the filename. This sets numIterations=1, requires SG_Mode to define the transport mode and overrides all other SG environment variables. The result will be a detailed impacts file for all zones to show the geographic effects of the single scenario run. This is DIFFERENT from the scenario generator batch impacts files, which are csv files showing aggregate impacts for all zones combined.


# Outputs
In the /data/outputs directory for DAFNI, or wherever defined for a local run:

calibration.yaml beta values and means trip lengths from a model calibration
debug.txt debugging information - not used except during testing
PyQUANT3_log.txt detailed logging information, including exception traces

One or other of these impacts files is generated, depending on the run type:
impacts_yyyyMMdd_hhmmss.csv time-stamped scenario impacts data from a batch of scenario runs
impacts_zones_yyyyMMdd_hhmmss.csv time-stamped zones impacts file from a single scenario run


# QUICK START - OPERATION OF PyQUANT3
The PyQUANT3 software can calibrate the model beta values, or run a scenario or batch of computer generated scenarios. This can be done on the user's machine by passing in command line switches, or on DAFNI by passing in the environment variables. The naming convention is the same for either method, so we just use the names of the variables in the following documentation.

USAGE 1 - CALIBRATION
=====================
OP_CODE=CALIBRATE (or pass any of the beta values in as zero)
This will calibrate the beta values for road, bus or rail, which are required to run the model.
Results are in the yaml file, which you then pass into the OPCODE=RUN options.

Result - 'calibration.yaml' file containing beta values for road, bus and rail

USAGE 2 - RUN A ONE OFF SCENARIO FROM A GRAPHML FILE
====================================================
OP_CODE=RUN
BetaRoad=0.1316692928026544
BetaBus=0.07288674278982177
BetaRail=0.06495053139819612
SG_MODE=0,1 or 2 - 0=road, 1=bus, 2=rail
SG_NETWORK=myscenario.graphml

The input graphml file contains the changes to be made to the network. These use lat/lon coordinates to define the origin and destination of links, but these are snapped to the nearest zone centroids.

There are examples of scenario graphmil files here: https://github.com/casa-ucl/QUANT-UDL-Scenarios/tree/master/QUANT2

The node section of the graphml file defines the lat/lon coordinates of the network nodes and an id which labels the node in the following edges section. The following is an example of both parts of the file:

<node id="CHH" code="" lines="E" lon="0.128262928" lat="51.56802985" name="Chadwell Heath Station"/>
and
<edge id="e0" source="CHH" target="XYZ">
		<data key="Weight">240</data>
</edge>

Here, the CHH node is defined at (0.12826,51.56803), which is then used as the source id in the nodes section to link this node to a different node labelled "XYZ". The weight of CHH to XYZ is the transit time in minutes for the node. Although using lat/lon coordinates to represent nodes, the QUANT model can only connect zones together, so the lat,lon is snapped to the closest zone centroid. This will be logged in the output file.


Result - 'impacts_zones_yyyyMMdd_hhmmss.csv'
The file format is a variant of 'docs/impacts.md' where each variable has one value per zone and each line in the file represents all the impacts for a single zone, identified by its MSOA area code and its zone i zone number.

USAGE 3 - RUN A SEQUENTIAL BATCH OF 1-LINK SCENARIOS
====================================================
OP_CODE=RUN
BetaRoad=0.1316692928026544
BetaBus=0.07288674278982177
BetaRail=0.06495053139819612
SG_NumIterations=50000
SG_Mode=0,1 or 2 - 0=road, 1=bus, 2=rail
SG_RadiusKM=5
SG_SpeedKPH=100
SG_Start_i=0
SG_Start_j=-1
SG_NumLinks=1

This runs an exhaustive scan of every zone, starting with i=start_i and j=start_j+1 which falls withing the RadiusKM distance between i and j. These are all single link scenarios, where the speed of the i-j link is changed to SpeedKPH and the QUANT model then run to compute the impacts.

Result - 'impacts_yyyyMMdd_hhmmss.csv' file contaiing the following aggregate variables with one line of the csv file per scenario run (i.e. there are SG_NumIterations rows in the file):
"idx,Ck1Road,Ck1Bus,Ck1Rail,Ck2Road,Ck2Bus,Ck2Rail,CkDiffRoad,CkDiffBus,CkDiffRail,"
"Lk1Road,Lk1Bus,Lk1Rail,Lk2Road,Lk2Bus,Lk2Rail,deltaLkRoad,deltaLkBus,deltaLkRail,"
"scenarioLinkDepthRoad, scenarioLinkDepthBus, scenarioLinkDepthRail,"
"scenarioLinkKMRoad, scenarioLinkKMBus, scenarioLinkKMRail,"
"scenarioLinkSavedSecsRoad, scenarioLinkSavedSecsBus, scenarioLinkSavedSecsRail,"
"LBarRoad, LBarBus, LBarRail,"
"nMinusRoad, nMinusBus, nMinusRail,"
"SavedSecsRoad, savedSecsBus, savedSecsRail"

The file format is explained in 'docs/impacts.md'.


USAGE 4 - RUN A BATCH OF RANDOM N-LINK SCENARIOS
================================================
OP_CODE=RUN
BetaRoad=0.1316692928026544
BetaBus=0.07288674278982177
BetaRail=0.06495053139819612
SG_NumIterations=50000
SG_Mode=0,1 or 2 - 0=road, 1=bus, 2=rail
SG_RadiusKM=5
SG_SpeedKPH=100
SG_NumLinks=2 (or anything >1)

When NumLinks>1, the code runs randomised computer generated scenarios using a different scenario generator from the exhaustive scan one used for NumLinks=1. This is because of the huge number of possible N link scenarios compared to the smaller number of 1 link scenarios, which can be scanned exhaustively. Given an approximate branching factor of 40 for 5KM links for the UK, this works out to approximately 175,000 1-link scenarios compared to about 13 million 2-link ones. That's why N-link is a randomised scenario generator.

Result - 'impacts_yyyyMMdd_hhmmss.csv' file contaiing the following aggregate variables with one line of the csv file per scenario run (i.e. there are SG_NumIterations rows in the file):
"idx,Ck1Road,Ck1Bus,Ck1Rail,Ck2Road,Ck2Bus,Ck2Rail,CkDiffRoad,CkDiffBus,CkDiffRail,"
"Lk1Road,Lk1Bus,Lk1Rail,Lk2Road,Lk2Bus,Lk2Rail,deltaLkRoad,deltaLkBus,deltaLkRail,"
"scenarioLinkDepthRoad, scenarioLinkDepthBus, scenarioLinkDepthRail,"
"scenarioLinkKMRoad, scenarioLinkKMBus, scenarioLinkKMRail,"
"scenarioLinkSavedSecsRoad, scenarioLinkSavedSecsBus, scenarioLinkSavedSecsRail,"
"LBarRoad, LBarBus, LBarRail,"
"nMinusRoad, nMinusBus, nMinusRail,"
"SavedSecsRoad, savedSecsBus, savedSecsRail"

The file format is explained in 'docs/impacts.md'. The only difference to 1-link is that the scenario data columns in the impacts file contain more data being comprised of n-links instead of 1-link. It's the same data, just repeated.

