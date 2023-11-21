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
import logging
from datetime import datetime
import yaml
import getopt
from pathlib import Path

#local imports
from utils import loadQUANTMatrix
from models.SingleOrigin import SingleOrigin

################################################################################

"""
parseArgs
@param argv List of command line args from the main program
This wouldn't normally be used in production, but it allows you to pass in
environment variables when debugging e.g. --opcode=CALIBRATE
"""
def parseArgs(argv):
    opts,args = getopt.getopt(argv, 'ho:', ['help','opcode='])
    for opt, arg in opts:
      if opt in('-h','--help'):
         print ('pyquant3.py -o [CALIBRATE|RUN]')
         sys.exit()
      elif opt in ('-o', '--opcode'):
         os.environ['OpCode']=arg

################################################################################

"""
main
@param argv command line arguments
Main program entry point. Reads config file and environment variables and
takes the action dictated by "OpCode" e.g. CALIBRATE or RUN
"""
def main(argv):

    print("hello world!")

    #set up files
    input_folder=Path("/data/inputs")
    output_folder=Path("/data/outputs")
    output_folder.mkdir(parents=True, exist_ok=True)
    log_file_name = output_folder.joinpath("PyQUANT3_log.txt")

    #start logging
    logging.basicConfig(filename=log_file_name, filemode='w', level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
    now = datetime.now()
    logging.info("PyQUANT3: starting run at "+now.strftime("%Y%m%d_%H:%M:%S"))


    #debugging preamble - check input folder for files and log
    output_file = output_folder.joinpath('debug.txt')
    try:
        output_file.write_text('different text')
        os.system('ls -l -R /data/inputs/model-runs >> /data/outputs/debug.txt') #real check of input files
        listdir_modelruns = str(os.listdir("/data/inputs/model-runs")) #inputs,outputs
        logging.info('os.listdir (model-runs): ' + listdir_modelruns)
    except Exception as e:
        logging.error("Exception: ", exc_info=True)
    ########


    #configuration file which contains names of all the files we need to run the model
    #these will have been mapped to /data/inputs/model-runs by DAFNI or our local launch config (tasks.json)
    #this MUST match the path set in appsettings.yaml "ModelRunsDir"
    configuration = yaml.safe_load(open("appsettings.yaml"))

    LocalModelRunsDir = configuration["dirs"]["ModelRunsDir"]
    ModelRunsDir = input_folder.joinpath(LocalModelRunsDir) #NOTE: this corrupts LocalModelRunsDir too!
    OutputDir = configuration["dirs"]["OutputDir"]
    TijObsRoadFilename = configuration["matrices"]["TObs1"]
    TijObsBusFilename = configuration["matrices"]["TObs2"]
    TijObsRailFilename = configuration["matrices"]["TObs3"]
    DisRoadFilename = configuration["matrices"]["dis_roads"]
    DisBusFilename = configuration["matrices"]["dis_buses"]
    DisGBRailFilename = configuration["matrices"]["dis_rail"]
    GreenBeltConstraintsFilename = configuration["tables"]["GreenBeltConstraints"]
    ConstraintsBFilename = configuration["tables"]["Constraints_B"]
    PopulationTableFilename = configuration["tables"]["PopulationArea"]
    ZoneCodesFilename = configuration["tables"]["ZoneCodes"]

    #now read any command line args which may override the previous
    parseArgs(argv)

    #inputs - this is to aid debugging as it will go into the console
    logging.info("pyquant3: LocalModelRunsDir = " + LocalModelRunsDir + " ModelRunsDir = " + str(ModelRunsDir))
    #prefix?
    logging.info("pyquant3: OutputDir = " + OutputDir)
    logging.info("pyquant3: TijObsRoadFilename = " + TijObsRoadFilename)
    logging.info("pyquant3: TijObsBusFilename = " + TijObsBusFilename)
    logging.info("pyquant3: TijObsGBRailFilename = " + TijObsRailFilename)
    logging.info("pyquant3: DisRoadFilename = " + DisRoadFilename)
    logging.info("pyquant3: DisBusFilename = " + DisBusFilename)
    logging.info("pyquant3: DisGBRailFilename = " + DisGBRailFilename)
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
        return

    
    #OK, so we've got a valid opcode and it's uppercase - we can continue
    logging.info("pyquant3: OpCode = " + opcode)

    logging.info("pyquant3: environment variables read, now loading data")

    #load matrices - from local files
    try:
        Tij_Obs_road = loadQUANTMatrix(os.path.join(ModelRunsDir,TijObsRoadFilename))
        Tij_Obs_bus = loadQUANTMatrix(os.path.join(ModelRunsDir,TijObsBusFilename))
        Tij_Obs_rail = loadQUANTMatrix(os.path.join(ModelRunsDir,TijObsRailFilename))
        #and costs
        Cij_road = loadQUANTMatrix(os.path.join(ModelRunsDir,DisRoadFilename))
        Cij_bus = loadQUANTMatrix(os.path.join(ModelRunsDir,DisBusFilename))
        Cij_rail = loadQUANTMatrix(os.path.join(ModelRunsDir,DisGBRailFilename))
    except Exception as e:
        logging.error("Exception in matrix loading: ", exc_info=True)
    ###


    #we have to add the all modes matrices because they don't exist in this form on the server
    #Tij_Obs = Tij_Obs_road + Tij_Obs_bus + Tij_Obs_rail #not needed?

    if opcode=='CALIBRATE':
        logging.info('calibrate')
        try:
            qm3 = SingleOrigin()
            qm3.TObs = [ Tij_Obs_road, Tij_Obs_bus, Tij_Obs_rail ]
            qm3.Cij = [ Cij_road, Cij_bus, Cij_rail ]
            qm3.isUsingConstraints=False
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
            #todo: need to write out predicted matrices here?
            #finished
            now = datetime.now()
            logging.info("PyQUANT3: finished run at "+now.strftime("%Y%m%d_%H:%M:%S"))
        except Exception as e:
            logging.error("Exception: ", exc_info=True)
    elif opcode=='RUN':
        logging.info('run')
        #todo: we need a changes file
    else:
        logging.error('Invalid OpCode: ' + opcode)

#END def main()

################################################################################

if __name__ == "__main__":
    main(sys.argv[1:])