"""
sweepcalibrate.py
Perform a calibration of QUANT by sweeping the beta values and comparing CBar and total population by modes
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
from scenarios.FileScenario import FileScenario
from scenarios.OneLink import OneLinkLimitR
from scenarios.NLink import NLinkLimitR

def sweep_calibrate(Tij_Obs_road, Tij_Obs_bus, Tij_Obs_rail, Cij_road, Cij_bus, Cij_rail):
    print("sweep calibrate\n")

    #global output_folder
    #global Tij_Obs_road, Tij_Obs_bus, Tij_Obs_rail
    #global Cij_road, Cij_bus, Cij_rail

    qm3 = SingleOrigin()
    qm3.TObs = [ Tij_Obs_road, Tij_Obs_bus, Tij_Obs_rail ]
    qm3.Cij = [ Cij_road, Cij_bus, Cij_rail ]
    #constraints initialisation - no constraints as default - need to initialise B weights to all 1.0
    qm3.isUsingConstraints=False
    (M, N) = np.shape(Tij_Obs_road)
    qm3.B = np.ones(N)
    #compute CijObs and CBarObs which don't change
    CijObs_road = np.sum(qm3.TObs[0])
    CijObs_bus = np.sum(qm3.TObs[1])
    CijObs_rail = np.sum(qm3.TObs[2])
    CBarObs_road = qm3.calculateCBar(qm3.TObs[0], qm3.Cij[0])
    CBarObs_bus = qm3.calculateCBar(qm3.TObs[1], qm3.Cij[1])
    CBarObs_rail = qm3.calculateCBar(qm3.TObs[2], qm3.Cij[2])


    with open('sweepcalibrate.csv','w') as file:
        file.write('beta_road,beta_bus,beta_rail,CijObs_road,CijObs_bus,CijObs_rail,Cij_road,Cij_bus,Cij_rail,CBarObs_road,CBarObs_bus,CBarObs_rail,CBarPred_road,CBarPred_bus,CBarPred_rail\n')
        for b1 in np.arange(0.02,0.2,0.01):
            for b2 in np.arange(0.02,0.2,0.01):
                for b3 in np.arange(0.02,0.2,0.01):
                    print(f"{b1},{b2},{b3}\n")
                    qm3.Beta = [b1,b2,b3]
                    #qm3.run() #NO!!!!!! This is the normal calibration which forces CBarObs errors to minimum
                    qm3.runWithChanges({}, None, False) #do a null run with no changes - takes obs mats and makes pred using betas
                    #and return the betas here
                    logging.info("beta (road)="+str(qm3.Beta[0])+" beta (bus)="+str(qm3.Beta[1])+" beta (rail)="+str(qm3.Beta[2]))
                    #the float() casts are because yaml.safe_dump can't handle the numpy double conversion properly "invalid object"
                    #calibration = {
                    #    'beta_road':float(qm3.Beta[0]), 'beta_bus':float(qm3.Beta[1]), 'beta_rail':float(qm3.Beta[2]),
                    #    'CBarObs_road':float(qm3.CBarObs[0]), 'CBarObs_bus':float(qm3.CBarObs[1]), 'CBarObs_rail':float(qm3.CBarObs[2]),
                    #    'CBarPred_road':float(qm3.CBarPred[0]), 'CBarPred_bus':float(qm3.CBarPred[1]), 'CBarPred_rail':float(qm3.CBarPred[2])
                    #}
                    Cij_road = np.sum(qm3.TPred[0])
                    Cij_bus = np.sum(qm3.TPred[1])
                    Cij_rail = np.sum(qm3.TPred[2])
                    #calculate mean predicted trips and mean observed trips (this is CBar)
                    #TODO: this is a bit of a hack as it's manipulating lots of things owned inside of qm3
                    qm3.CBarPred = [0.0 for k in range(0,qm3.numModes)]
                    qm3.CBarObs = [0.0 for k in range(0,qm3.numModes)]
                    delta = [0.0 for k in range(0,qm3.numModes)]
                    for k in range(0,qm3.numModes):
                        qm3.CBarPred[k] = qm3.calculateCBar(qm3.TPred[k], qm3.Cij[k])
                        #qm3.CBarObs[k] = qm3.calculateCBar(qm3.TObs[k], qm3.Cij[k])
                        #print("Mode "+str(k)+" CBar Pred="+str(self.CBarPred[k])+" CBar Obs="+str(self.CBarObs[k]))
                        delta[k] = np.fabs(qm3.CBarPred[k] - qm3.CBarObs[k])
                    #end for k
                    print(f"{b1},{b2},{b3},{qm3.CBarPred[0]},{qm3.CBarPred[1]},{qm3.CBarPred[2]}\n")
                    file.write( f"{b1},{b2},{b3},"
                          +f"{CijObs_road},{CijObs_bus},{CijObs_rail},"
                          +f"{Cij_road},{Cij_bus},{Cij_rail},"
                          +f"{CBarObs_road},{CBarObs_bus},{CBarObs_rail},"
                          +f"{qm3.CBarPred[0]},{qm3.CBarPred[1]},{qm3.CBarPred[2]}\n"
                    )
                    file.flush()
                #end for b3
            #end for b2
        #end for b1
    #end open
#end def

