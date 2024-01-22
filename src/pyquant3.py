#!/usr/bin/python

"""
PyQUANT3
Python version of QUANT3 model to run on DAFNI (https://dafni.ac.uk)

Inputs (Environment Variables):
OpCode = CALIBRATE | RUN
Beta?

How it works Docker-wise
NOTE: tasks.json maps local model-runs to /data in container
NOTE: data goes into /data/outputs in order to be read out of the container
NOTE: tasks.json also contains an "env" property where we're setting our OpCode

Uploading to DAFNI
docker build -t pyquant3:latest .
docker save -o pyquant3.tar pyquant3:latest
then gzip it

Running locally on Docker (NOTE: vscode does this, but add inputs too - it's in the tasks.json)
docker run --mount type=bind,source="C:/richard/github/pyquant3/outputs",target="/data/outputs" pyquant3:latest

CuPy JAX?

===

DAFNI VERSION GUIDs for datasets:
TObs1 - 61ba53be-42c4-40de-88fe-326d115f0f5d
TObs2 - a9e9e88d-b422-4d5a-a454-8abf56587c98
TObs3 - 9d9d20cc-d901-494f-8ae6-46a045f284f9
RoadCosts - 0faecc27-7974-4d97-b11b-5a661ba5322d
BusCosts - 072ead5b-2c5b-4877-b2cf-d61f6acef73e
RailCosts - 0c9a8a60-4ecd-4893-bb92-2bdac180e820
ZoneCodes - d5faa9b9-8a4a-421c-a0bd-cd279ab2aa68 (not needed?)

"""

#libraries
import sys
import os
import pickle
import copy
import logging
from datetime import datetime
import time
import yaml
import getopt
from pathlib import Path
import numpy as np
import pandas as pd

#local imports
from utils import loadQUANTMatrix, loadQUANTMatrixFAST
from models.SingleOrigin import SingleOrigin
from models.DirectNetworkChange import DirectNetworkChange
from impacts.ImpactStatistics import ImpactStatistics
from networks.NetworkUtils import NetworkUtils
from scenarios.OneLink import OneLinkLimitR

################################################################################

"""
parseArgs
@param argv List of command line args from the main program
This wouldn't normally be used in production, but it allows you to pass in
environment variables when debugging e.g. --opcode=CALIBRATE
"""
def parseArgs(argv):
    opts,args = getopt.getopt(argv,
            'hdo:i:j:',
            ['help','dafni','opcode=','betaroad=','betabus=','betarail=','numiterations=','mode=','radiuskm=','speedkph=','starti=','startj='])
    for opt, arg in opts:
        if opt in('-h','--help'):
            print ('pyquant3.py -o [CALIBRATE|RUN] [--betaroad] [--betabus] [--beta rail]')
            print('[--numiterations=10] [--radiuskm=5] [--mode=2] [--speedkph=100] [-i 0 | --starti=0] [-j 0 | --startj=0]')
            sys.exit()
        elif opt in ('-d','--dafni'):
            os.environ['IsOnDAFNI']=True
        elif opt in ('-o', '--opcode'):
            os.environ['OpCode']=arg
        elif opt in ('--betaroad'):
            os.environ['BetaRoad']=arg
        elif opt in ('--betabus'):
            os.environ['BetaBus']=arg
        elif opt in ('--betarail'):
            os.environ['BetaRail']=arg
        elif opt in ('--numiterations'):
            os.environ['SG_NumIterations']=arg
        elif opt in ('--mode'):
            os.environ['SG_Mode']=arg
        elif opt in ('--radiuskm'):
            os.environ['SG_RadiusKM']=arg
        elif opt in ('--speedkph'):
            os.environ['SG_SpeedKPH']=arg
        elif opt in ('-i','--starti'):
            os.environ['SG_Start_i']=arg
        elif opt in ('-j','--startj'):
            os.environ['SG_Start_j']=arg
#end def

################################################################################

"""
main
@param argv command line arguments
Main program entry point. Reads config file and environment variables and
takes the action dictated by "OpCode" e.g. CALIBRATE or RUN
"""
def main(argv):
    global input_folder, output_folder
    global Tij_Obs_road, Tij_Obs_bus, Tij_Obs_rail #observed flows between zones
    global Cij_road, Cij_bus, Cij_rail #costs (time minutes) between zones
    global Lij_road, Lij_bus, Lij_rail #distance between zones

    print("hello world!")

    #read any command line args which may override the following file paths
    parseArgs(argv)

    #configuration file which contains names of all the files we need to run the model
    #these will have been mapped to /data/inputs/model-runs by DAFNI or our local launch config (tasks.json)
    #this MUST match the path set in appsettings.yaml "ModelRunsDir"
    configuration = yaml.safe_load(open("appsettings.yaml"))

    #set up files - depending if we're running on DAFNI or locally
    isOnDAFNI = os.getenv("IsOnDAFNI",False)

    ########
    #debugging preamble for DAFNI - check input folder for files and log
    if isOnDAFNI:
        try:
            os.system('ls -l -R /data/inputs/model-runs >> /data/outputs/debug.txt') #real check of input files
            os.system('set >> /data/outputs/debug.txt') #list envonment variables
        except Exception as e:
            #NOTE: we can't log here as there is no log file yet
            print(e)
    ########

    if isOnDAFNI:
        #DAFNI input_folder=Path("/data/inputs")
        #DAFNI output_folder=Path("/data/outputs")
        input_folder = Path(configuration["dirs"]["DAFNIInputsDir"])
        output_folder = Path(configuration["dirs"]["DAFNIOutputsDir"])
        ModelRunsDirName = configuration["dirs"]["DAFNIModelRunsDir"]
    else:
        input_folder=Path(configuration["dirs"]["LocalInputsDir"])
        output_folder=Path(configuration["dirs"]["LocalOutputsDir"])
        ModelRunsDirName = configuration["dirs"]["LocalModelRunsDir"]

    #now we can start the logging as we know where to put the file
    output_folder.mkdir(parents=True, exist_ok=True)
    log_file_name = output_folder.joinpath("PyQUANT3_log.txt")

    #start logging
    logging.basicConfig(filename=log_file_name, filemode='w', level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
    now = datetime.now()
    logging.info("PyQUANT3: starting run at "+now.strftime("%Y%m%d_%H:%M:%S"))
    logging.info("PyQUANT3: IsOnDAFNI=" + str(isOnDAFNI))

    ModelRunsDir = input_folder.joinpath(ModelRunsDirName) #NOTE: this corrupts input folder????
    TijObsRoadFilename = configuration["matrices"]["TObs1"]
    TijObsBusFilename = configuration["matrices"]["TObs2"]
    TijObsRailFilename = configuration["matrices"]["TObs3"]
    DisRoadFilename = configuration["matrices"]["dis_roads"]
    DisBusFilename = configuration["matrices"]["dis_buses"]
    DisGBRailFilename = configuration["matrices"]["dis_rail"]
    DisCrowflyKMFilename = configuration["matrices"]["dis_crowfly"]
    DisCrowflyVertexRoadsKMFilename = configuration["matrices"]["dis_crowfly_vertex_roads_KM"]
    DisCrowflyVertexBusKMFilename = configuration["matrices"]["dis_crowfly_vertex_bus_KM"]
    DisCrowflyVertexGBRailKMFilename = configuration["matrices"]["dis_crowfly_vertex_gbrail_KM"]
    GreenBeltConstraintsFilename = configuration["tables"]["GreenBeltConstraints"]
    ConstraintsBFilename = configuration["tables"]["Constraints_B"]
    PopulationTableFilename = configuration["tables"]["PopulationArea"]
    ZoneCodesFilename = configuration["tables"]["ZoneCodes"]

    #inputs - this is to aid debugging as it will go into the console
    logging.info("pyquant3: ModelRunsDirName = " + ModelRunsDirName + " ModelRunsDir = " + str(ModelRunsDir))
    #prefix?
    logging.info("pyquant3: TijObsRoadFilename = " + TijObsRoadFilename)
    logging.info("pyquant3: TijObsBusFilename = " + TijObsBusFilename)
    logging.info("pyquant3: TijObsGBRailFilename = " + TijObsRailFilename)
    logging.info("pyquant3: DisRoadFilename = " + DisRoadFilename)
    logging.info("pyquant3: DisBusFilename = " + DisBusFilename)
    logging.info("pyquant3: DisGBRailFilename = " + DisGBRailFilename)
    logging.info("pyquant3: DisCrowflyKMFilename = " + DisCrowflyKMFilename)
    logging.info("pyquant3: DisCrowflyVertexRoadsKMFilename = " + DisCrowflyVertexRoadsKMFilename)
    logging.info("pyquant3: DisCrowflyVertexBusKMFilename = " + DisCrowflyVertexBusKMFilename)
    logging.info("pyquant3: DisCrowflyVertexGBRailKMFilename = " + DisCrowflyVertexGBRailKMFilename)
    logging.info("pyquant3: GreenBeltConstraintsFilename = " + GreenBeltConstraintsFilename)
    logging.info("pyquant3: ConstraintsBFilename = " + ConstraintsBFilename)
    logging.info("pyquant3: PopulationTableFilename = " + PopulationTableFilename)
    logging.info("pyquant3: ZoneCodesFilename = " + ZoneCodesFilename)


    #Environment variables

    #operation code
    opcode = os.getenv("OpCode")
    if opcode:
        opcode = opcode.upper() #calibrate, run etc. TODO: implement
    else:
        logging.error("pyquant3: ERROR: no opcode passed in environment variables, exiting")
        print("ERROR: no opcode passed in environment variables, exiting")
        return

    
    #OK, so we've got a valid opcode and it's uppercase - we can continue
    logging.info("pyquant3: OpCode = " + opcode)

    logging.info("pyquant3: environment variables read, now loading data")

    #load matrices - from local files
    try:
        Tij_Obs_road = loadQUANTMatrixFAST(os.path.join(ModelRunsDir,TijObsRoadFilename))
        Tij_Obs_bus = loadQUANTMatrixFAST(os.path.join(ModelRunsDir,TijObsBusFilename))
        Tij_Obs_rail = loadQUANTMatrixFAST(os.path.join(ModelRunsDir,TijObsRailFilename))
        #and costs
        Cij_road = loadQUANTMatrixFAST(os.path.join(ModelRunsDir,DisRoadFilename))
        Cij_bus = loadQUANTMatrixFAST(os.path.join(ModelRunsDir,DisBusFilename))
        Cij_rail = loadQUANTMatrixFAST(os.path.join(ModelRunsDir,DisGBRailFilename))
        #and transport KM distances
        Lij_road = loadQUANTMatrixFAST(os.path.join(ModelRunsDir,DisCrowflyVertexRoadsKMFilename))
        Lij_bus = loadQUANTMatrixFAST(os.path.join(ModelRunsDir,DisCrowflyVertexBusKMFilename))
        Lij_rail = loadQUANTMatrixFAST(os.path.join(ModelRunsDir,DisCrowflyVertexGBRailKMFilename))
    except Exception as e:
        logging.error("Exception in matrix loading: ", exc_info=True)
        print(e)
    ###

    #load zone codes xml file into a pandas dataframe
    try:
        with open(os.path.join(ModelRunsDir,ZoneCodesFilename)) as f:
            df_ZoneCodes = pd.read_xml(f)
            print(df_ZoneCodes.head(10))
            print("ZoneCodes has "+str(len(df_ZoneCodes.index))+" rows")
    except Exception as e:
        logging.error("Exception loading zone codes xml file: ", exc_info=True)
        print(e)


    #we have to add the all modes matrices because they don't exist in this form on the server
    #Tij_Obs = Tij_Obs_road + Tij_Obs_bus + Tij_Obs_rail #not needed?

    if opcode=='CALIBRATE':
        logging.info('calibrate')
        try:
            calibrate(0.0,0.0,0.0) #pass in zeros for betas to trigger a full calibration to find the real betas
            #todo: need to write out predicted matrices here?
            #finished
            #now = datetime.now()
            #logging.info("PyQUANT3: finished run at "+now.strftime("%Y%m%d_%H:%M:%S"))
        except Exception as e:
            logging.error("Exception: ", exc_info=True)
            print(e)
    elif opcode=='RUN':
        #todo: we need a changes file
        #todo: or we could assume that it's passing in a code to make a randomised scenario?
        #basic plan: calibrate (or hard code), make up a scenario, run scenario, measure impacts [repeat]
        logging.info('run')

        #load and check for SG (scenario generator) environment variables
        numIterations = int(os.getenv('SG_NumIterations','10'))
        mode = int(os.getenv('SG_Mode','0'))
        radiusKM = float(os.getenv('SG_RadiusKM','5.0'))
        speedKPH = float(os.getenv('SG_SpeedKPH','100.0'))
        start_i = int(os.getenv('SG_Start_i','0'))
        start_j = int(os.getenv('SG_Start_j','-1'))
        logging.info('SG_NumIterations='+str(numIterations))
        logging.info('SG_Mode='+str(mode))
        logging.info('SG_RadiusKM='+str(radiusKM))
        logging.info('SG_SpeedKPH='+str(speedKPH))
        logging.info('SG_Start_i='+str(start_i))
        logging.info('SG_Start_j='+str(start_j))

        #start an impacts file here
        now = datetime.now()
        impacts_file = output_folder.joinpath("impacts_"+now.strftime("%Y%m%d_%H%M%S")+".csv")

        try:
            with impacts_file.open('w') as f: #open an impacts log file here...
                #look for betas in the environment variables, which lets us skip the lengthy calibration stage
                betaRoad = float(os.getenv("BetaRoad", default='0.0'))
                betaBus = float(os.getenv("BetaBus", default='0.0'))
                betaRail = float(os.getenv("BetaRail", default='0.0'))
            
                qm3_base = calibrate(betaRoad,betaBus,betaRail) #calibrate our model - only if no betas passed in
                #this was used if you want to save the whole baseline model object for later - 5GB! doesn't work on DAFNI
                #with open('outputs/qm3_base.bin', 'wb') as qfile: #todo: it's [output_folder]/qm3_base.bin on DAFNI
                #    pickle.dump(qm3_base, qfile)
                saveCij = [ np.copy(qm3_base.Cij[k]) for k in range(0,qm3_base.numModes)] #save the pre-scenario Cij matrix as we're about to change it

                #write the header line to the impacts file
                f.write(
                    "idx,"
                    +"Ck1Road,Ck1Bus,Ck1Rail,Ck2Road,Ck2Bus,Ck2Rail,CkDiffRoad,CkDiffBus,CkDiffRail,"
                    +"Lk1Road,Lk1Bus,Lk1Rail,Lk2Road,Lk2Bus,Lk2Rail,deltaLkRoad,deltaLkBus,deltaLkRail,"
                    +"scenarioLinkDepthRoad, scenarioLinkDepthBus, scenarioLinkDepthRail,"
                    +"scenarioLinkKMRoad, scenarioLinkKMBus, scenarioLinkKMRail,"
                    +"scenarioLinkSavedSecsRoad, scenarioLinkSavedSecsBus, scenarioLinkSavedSecsRail,"
                    +"LBarRoad, LBarBus, LBarRail,"
                    +"nMinusRoad, nMinusBus, nMinusRail,"
                    +"SavedSecsRoad, savedSecsBus, savedSecsRail,"
                    +"net_mode, net_i, net_j, net_secs\n"
                )
                N = len(df_ZoneCodes.index)
                #linkSpeed = speedKPH #KPH
                scenarioGenerator = OneLinkLimitR(radiusKM,N,mode,Lij_rail) #was 20KM, not 5
                scenarioGenerator.i=start_i #carry on where we left off
                scenarioGenerator.j=start_j #carry on where we left off

                for i in range(0,numIterations):
                    print('iteration '+str(i))
                    now = datetime.now()
                    logging.info('Iteration '+str(i)+' start: '+now.strftime("%Y%m%d_%H%M%S"))
                    
                    start_time=time.process_time()
                    #qm3 = qm3_base.deepcopy() #clone a new QUANT model so we can apply changes and difference with the baseline
                    #with open('outputs/qm3_base.bin', 'rb') as qfile:
                    #    qm3 = pickle.load(qfile)
                    qm3 = copy.copy(qm3_base) #don't deep clone the whole model, just copy Cij - nothing else changes (I hope!)
                    qm3.Cij = [ np.copy(saveCij[k]) for k in range(0,qm3.numModes) ] #copy pre-scenario Cij back
                    #OK, so what we've jsut done is to shallow copy qm3_base, then write over the Cij properties
                    #so that we're free to change them in qm3 for the new scenario, while being able to get the
                    #original Cij back quickly again from saveCij - only Cij is changed by the scenario
                    #NOTE: we need qm3_base to hold the unchanged Cij when we go to the impacts, otherwise
                    #you can't compute how many route changes have been made
                    #This method takes around 1s, comapred to the 10s of the full qm3=deepcopy(qm3_base) method
                    end_time = time.process_time()
                    print('pyquant3::deep_copy '+str(end_time-start_time)+' secs')
                    logging.info('pyquant3::deep_copy '+str(end_time-start_time)+' secs')

                    ###scenario changes section here - make up a scenario
                    OiDjHash = {} #hash of zonei number as key, with array [Oi,Dj] new totals as value
                    #todo: you need to make up some network changes here
                    #r = NetworkUtils.linkKMPerHourToSeconds(0,1,Lij_rail,100.0) #0->1 at 100KPH
                    #networkChanges = {
                    #    DirectNetworkChange(2,0,1,r)  #mode=2,i=0,j=1,r=runlink in seconds
                    #}
                    networkChanges = scenarioGenerator.next()
                    for nc in networkChanges:
                        r = NetworkUtils.linkKMPerHourToSeconds(nc.originZonei,nc.destinationZonei,Lij_rail,speedKPH)
                        nc.absoluteTimeSecs = r
                    ###end of scenario changes section

                    ###scenario run section
                    start_time = time.process_time()
                    #NOTE: runWithChanges will alter dis matrices - just in case you're doing multiple runs
                    qm3.runWithChanges(OiDjHash,networkChanges,False)
                    end_time = time.process_time()
                    print('pyquant3:: qm3.runWithChanges() '+str(end_time-start_time)+' secs')
                    logging.info('pyquant3:: qm3.runWithChanges() '+str(end_time-start_time)+' secs')
                    ###end scenario run section

                    now = datetime.now()
                    logging.info('Iteration '+str(i)+' model finished, starting impact statistics: '+now.strftime("%Y%m%d_%H%M%S"))

                    ###write out impacts section
                    #now output results - impacts - score?
                    start_time = time.process_time()
                    impacts = ImpactStatistics()
                    impacts.compute(qm3_base,qm3,[ Lij_road, Lij_bus, Lij_rail ], networkChanges)
                    end_time = time.process_time()
                    print('pyquant3:: impacts.compute '+str(end_time-start_time)+' secs')
                    logging.info('pyquant3:: impacts.compute '+str(end_time-start_time)+' secs')
                    f.write(
                        ('{0},' #idx
                        '{1},{2},{3},' #Ck1
                        '{4},{5},{6},' #Ck2
                        '{7},{8},{9},' #CkDiff
                        '{10},{11},{12},' #Lk1
                        '{13},{14},{15},' #Lk2
                        '{16},{17},{18},' #deltaLk
                        '{19},{20},{21},' #scenarioLinkDepth_k
                        '{22},{23},{24},' #scenarioLinkKM_K
                        '{25},{26},{27},' #scenarioLinkSavedSecs_K
                        '{28},{29},{30},' #LBar_k
                        '{31},{32},{33},' #nMinus_K
                        '{34},{35},{36}' #savedSecs_K
                        )
                        .format(
                            i,
                            impacts.Ck1[0],impacts.Ck1[1],impacts.Ck1[2],
                            impacts.Ck2[0],impacts.Ck2[1],impacts.Ck2[2],
                            impacts.CkDiff[0],impacts.CkDiff[1],impacts.CkDiff[2],
                            impacts.Lk1[0],impacts.Lk1[1],impacts.Lk1[2],
                            impacts.Lk2[0],impacts.Lk2[1],impacts.Lk2[2],
                            impacts.deltaLk[0],impacts.deltaLk[1],impacts.deltaLk[2],
                            impacts.scenarioLinkDepth_k[0], impacts.scenarioLinkDepth_k[1], impacts.scenarioLinkDepth_k[2],
                            impacts.scenarioLinkKM_k[0], impacts.scenarioLinkKM_k[1], impacts.scenarioLinkKM_k[2],
                            impacts.scenarioLinkSavedSecs_k[0], impacts.scenarioLinkSavedSecs_k[1], impacts.scenarioLinkSavedSecs_k[2],
                            impacts.LBar_k[0], impacts.LBar_k[1], impacts.LBar_k[2],
                            impacts.nMinus_k[0], impacts.nMinus_k[1], impacts.nMinus_k[2],
                            impacts.savedSecs_k[0], impacts.savedSecs_k[1], impacts.savedSecs_k[2]
                        )
                    )
                    #now write out the game state: mode,i,j,secs for each network change
                    for nc in networkChanges:
                        f.write(',{0},{1},{2},{3}'.format(nc.mode,nc.originZonei,nc.destinationZonei,nc.absoluteTimeSecs))
                    f.write('\n')
                    f.flush()

                    #this is what QUANT3 AI does
                    #writer.Write("idx,score,depth,combs,netChgRoad,netChgBus,netChgRail,"
                    # + "netSavedMinsRoad,netSavedMinsBus,netSavedMinsRail,"
                    # + "savedMinsRoad,savedMinsBus,savedMinsRail,"
                    # +"deltaDkRoad,deltaDkBus,deltaDkRail,"
                    # +"deltaLkRoad,deltaLkBus,deltaLkRail,"+modeText+"NetworkKM,LBar,"
                    # +"ATI,ATIPop");

                    now = datetime.now()
                    logging.info('Iteration '+str(i)+' finish: '+now.strftime("%Y%m%d_%H%M%S"))

                #end for i
            #end with f
        except Exception as e:
            logging.error("Exception: ", exc_info=True)
            print(e)
    else:
        logging.error('Invalid OpCode: ' + opcode)
        print('Invalid OpCode: ' + opcode)

    #finished
    now = datetime.now()
    logging.info("PyQUANT3: finished at "+now.strftime("%Y%m%d_%H:%M:%S"))

#END def main()

################################################################################

"""
Calibrate and write out yaml file
@param "betaRoad" road beta - if all three passed in then calibration skipped
@param "betaBus" bus beta - if all three passed in then calibration skipped
@param "betaRail" rail beta - if all three passed in then calibration skipped
@returns QuantModel3 which is now calibrated
"""
def calibrate(betaRoad,betaBus,betaRail):
    global output_folder
    global Tij_Obs_road, Tij_Obs_bus, Tij_Obs_rail
    global Cij_road, Cij_bus, Cij_rail

    qm3 = SingleOrigin()
    qm3.TObs = [ Tij_Obs_road, Tij_Obs_bus, Tij_Obs_rail ]
    qm3.Cij = [ Cij_road, Cij_bus, Cij_rail ]
    #constraints initialisation - no constraints as default - need to initialise B weights to all 1.0
    qm3.isUsingConstraints=False
    (M, N) = np.shape(Tij_Obs_road)
    qm3.B = np.ones(N)
    
    #skip calibration stage if non valid betas have been passed in - useful for scenario runs
    if betaRoad>0 and betaBus>0 and betaRail>0:
        qm3.Beta=[betaRoad,betaBus,betaRail]
        logging.info("Calibration skipped as betas passed in environment: "+str(betaRoad)+" "+str(betaBus)+" "+str(betaRail))
        qm3.fastComputePredicted() #computes a baseline TPred set from the betas
    else:
        logging.info("Calibration from matrices as no betas passed in environment")
        qm3.run()
        #and return the betas here
        logging.info("beta (road)="+str(qm3.Beta[0])+" beta (bus)="+str(qm3.Beta[1])+" beta (rail)="+str(qm3.Beta[2]))
        #the float() casts are because yaml.safe_dump can't handle the numpy double conversion properly "invalid object"
        calibration = {
            'beta_road':float(qm3.Beta[0]), 'beta_bus':float(qm3.Beta[1]), 'beta_rail':float(qm3.Beta[2]),
            'CBarObs_road':float(qm3.CBarObs[0]), 'CBarObs_bus':float(qm3.CBarObs[1]), 'CBarObs_rail':float(qm3.CBarObs[2]),
            'CBarPred_road':float(qm3.CBarPred[0]), 'CBarPred_bus':float(qm3.CBarPred[1]), 'CBarPred_rail':float(qm3.CBarPred[2])
        }
        yaml.safe_dump(calibration, open(output_folder.joinpath('calibration.yaml'),'w'))
    #end if

    return qm3
#end def calibrate


################################################################################

if __name__ == "__main__":
    main(sys.argv[1:])