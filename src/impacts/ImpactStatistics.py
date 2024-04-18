"""
ImpactStatistics.py
Compute impact statistics for a set of matrices from a scenario run
"""

#from numba import jit
#from numba.experimental import jitclass
#from numba.typed import List
from numba import float64, types, typed
#from typing import List
#from numba.typed import List as NumbaList
import numpy as np
from jax import grad, jit
from jax import lax
from jax import random
import jax
import jax.numpy as jnp
#import cupy as cp
import typing as pt

from models.DirectNetworkChange import DirectNetworkChange
from models.SingleOrigin import SingleOrigin

#spec = [
#    ('Ck1', float64[:]),
#    ('Ck2', float64[:]),
#    ('CkDiff', float64[:]),
#    ('Lk1', float64[:]),
#    ('Lk2', float64[:]),
#    ('deltaLk', float64[:]),
#    ('scenarioLinkDepth_k', float64[:]),
#    ('scenarioLinkKM_k', float64[:]),
#    ('scenarioLinkSavedSecs_k', float64[:]),
#    ('LBar_k', float64[:]),
#    ('nMinus_k', float64[:]),
#    ('savedSecs_k', float64[:])
#]

#@jitclass()
class ImpactStatistics(object):
    Ck1: pt.List[float64]
    Ck2: pt.List[float64]
    CkDiff: pt.List[float64]
    Lk1: pt.List[float64]
    Lk2: pt.List[float64]
    deltaLk: pt.List[float64]
    scenarioLinkDepth_k: pt.List[float64]
    scenarioLinkKM_k: pt.List[float64]
    scenarioLinkSavedSecs_k: pt.List[float64]
    LBar_k: pt.List[float64]
    nMinus_k: pt.List[float64]
    savedSecs_k: pt.List[float64]

    def __init__(self):
        self.Ck1 = typed.List([0.0]) #baseline population count (people) by mode
        self.Ck2 = typed.List([0.0]) #scenario population count (people) by mode
        self.CkDiff = typed.List([0.0]) #Ck2-Ck1 scenario minus baseline
        #self.deltaDk = []
        self.Lk1 = typed.List([0.0]) #baseline distance (KM) travelled on each mode
        self.Lk2 = typed.List([0.0]) #scenario distance (KM) travelled on each mode
        self.deltaLk = typed.List([0.0]) #Lk2=Lk1 difference in distance travelled by mode

        #scenario link network changes - these are physical changes made by the scenario
        self.scenarioLinkDepth_k = typed.List([0.0]) #this is how many changed links there are on each mode
        self.scenarioLinkKM_k = typed.List([0.0]) #distances of added links by mode (KM)
        self.scenarioLinkSavedSecs_k = typed.List([0.0]) #times saved by new added linsk by mode (secs)
        self.LBar_k = typed.List([0.0]) #average geographic spread in KM of scenario nodes

        #scenario network changes - these are results of the APSP algorithm on network structure
        self.nMinus_k = typed.List([0.0])
        self.savedSecs_k = typed.List([0.0])

        #now the vector ones - these are all [k,i] or [k,j]
        self.Cik1 = np.array((1,1),dtype=float)
        self.Cik2 = np.array((1,1),dtype=float)
        self.CikDiff = np.array((1,1),dtype=float)
        self.Cjk1 = np.array((1,1),dtype=float)
        self.Cjk2 = np.array((1,1),dtype=float)
        self.CjkDiff = np.array((1,1),dtype=float)
    
################################################################################

    """
    Compute total distance travelled on each mode
    @param Tij FMatrix[] trips matrix
    @param dijKM FMatrix[] distance matrix
    @returns double [] total distance travelled by people
    """
    #@jit(nopython=True)
    def computeL_k(self, Tij: np.matrix, dijKM: np.matrix) -> list[float]:
        NumModes = len(Tij)
        (M, N) = np.shape(Tij[0])
        Lk = [0.0 for k in range(0,NumModes)]
        for k in range(0, NumModes):
            #Lk[k] = 0
            #for i in range(0,N):
            #    Sum = 0.0
            #    for j in range(0,N):
            #        Sum += Tij[k][i, j] * dijKM[k][i, j]
            #    Lk[k] += Sum
            #faster - numpy
            #Lk[k]=np.sum(Tij[k] * dijKM[k])
            #JAX
            Lk[k]=jnp.sum(Tij[k] * dijKM[k])
        return Lk

################################################################################

    """
    computeScenarioLinkStatistics
    Compute all the statistics related to the network scenario geometry that has
    been added to make the scenario i.e. distances between the links we've added
    and the number of primary seconds that this saves (all per mode)
    NOTE: we use the dijKM matrix here, which is distances between network
    vertices. This is different from using zone centroids from ZoneCodes. This is
    probably the better method as it's consistent with the network distances.
    @param networkChanges List(DirectNetworkChange) The scenario changes telling us which links to connect
    @param Cij The time in minutes between each zone in the model by mode
    @param dijKM The distances (KM) between each zone in the model by mode
    @returns double[3] scenarioLinkDepth, double[3] KM between link and double[3] minutes saved
    """
    #@jit(nopython=True)
    def computeScenarioLinkStatistics(self, networkChanges, Cij, dijKM):
        NumModes = len(dijKM)
        scenarioLinkDepth_k = [0 for k in range(0,NumModes)]
        scenarioLinkSavedSecs_k = [0.0 for k in range(0,NumModes)]
        scenarioLinkKM_k = [0.0 for k in range(0,NumModes)]
        for nc in networkChanges:
            k = nc.mode
            i = nc.originZonei
            j = nc.destinationZonei
            link_secs = nc.absoluteTimeSecs
            distKM = dijKM[k][i,j]
            scenarioLinkDepth_k[k]+=1 #count number of changes on each mode
            #note Cij in minutes and network run lengths in seconds
            savedSecs = max(Cij[k][i,j]*60-link_secs,0) #existing link time - new link time, but lower limit to zero
            scenarioLinkSavedSecs_k[k]+=savedSecs
            scenarioLinkKM_k[k]+=distKM
        #end for
        return scenarioLinkDepth_k, scenarioLinkKM_k, scenarioLinkSavedSecs_k

################################################################################

    """
    computeLBar
    Compute how far apart the network changes are for each mode individually.
    Basically, you make a list of every origin and destination node, then
    take every combination of every node to every other node and sum the
    distance. Then divide by the number of combinations and that's a measure
    of the geographic spread of the mode changes. Repeat for each mode.
    @param networkChanges The scenario (list of DirectNetworkChange)
    @param dijKM_k distance matrix in km with all three modes (the crowfly vertex KM one)
    @returns average distance between scenario changes nodes (NOTE: =0 if there are no changes on a mode)
    """
    #@jit(nopython=True)
    def computeLBar(self, networkChanges, dijKM_k):
        NumModes = len(dijKM_k)
        LBar_k = [0.0 for k in range(0,NumModes)]
        
        for k in range(0,NumModes):
            nodes = {}
            for nc in networkChanges:
                if nc.mode==k:
                    i = nc.originZonei
                    j = nc.destinationZonei
                    if not i in nodes:
                        nodes[i]=True # no hashset, so we're using a hash map - values are irrelevant
                    if not j in nodes:
                        nodes[j]=True
                #end if k=mode
            #end for nc

            #OK, we've got a list of nodes, we now need to permute every combination
            #and add up the distances
            sum=0.0
            count=0
            for i in nodes:
                for j in nodes:
                    if i!=j:
                        count+=1
                        sum+=dijKM_k[k][i,j]
                    #end if
                #end for j
            #end for i
            if count>0:
                LBar_k[k] = sum/count
            else:
                LBar_k[k]=0
        #end for k
        return LBar_k

################################################################################

    """
    computeNetworkStatistics
    This is derived network statistics (secondary) as a result of the primary
    changes from the network scenario links that are added.
    nMinus - number of (geometric) trips that are now quicker
    savedSecs - number of seconds saved by the new shortest path geometry
    @param Cij1 baseline shortest paths (minutes) matrix by mode
    @param Cij2 scenario shortest paths (minutes) matrix by mode
    @returns nMinus, savedSecs
    """
    #@jit(nopython=True)
    def computeNetworkStatistics(self,Cij1,Cij2):
        NumModes = len(Cij1)
        nMinus_k = [0.0 for k in range(0,NumModes)]
        savedSecs_k = [0.0 for k in range(0,NumModes)]
        for k in range(0,NumModes):
            nMinus_k[k]=jnp.count_nonzero(Cij2[k] < Cij1[k])
            diff = Cij1[k]-Cij2[k] #it's saved seconds, and 2<1 if you're saving secs and it's quicker
            diff = jnp.where(diff>0,diff,0) #filter out any negative values - savings are all positive
            savedSecs_k[k] =np.sum(diff)
        #end for
        return nMinus_k, savedSecs_k


################################################################################

    """
    compute
    Computes impact statics given a baseline model and a scenario changes model
    @param qm3_base baseline quant 3 model
    @param qm3 scenario quant 3 model
    @param dijKM vertex KM distance file, 3 modes
    @param networkChanges The scenario changes so we can compute distances, spread, geographic statistics etc.
    """
    #@jit(nopython=True)
    def compute(self,qm3_base: SingleOrigin,qm3: SingleOrigin, dijKM:np.matrix, networkChanges: list):
        #print("DNC::",type(networkChanges))

        (M, N) = np.shape(qm3.TObs[0])

        #count people on modes
        self.Ck1 = [0 for k in range(0,qm3.numModes)]
        self.Ck2 = [0 for k in range(0,qm3.numModes)]
        self.CkDiff = [0 for k in range(0,qm3.numModes)]
        for k in range(0,qm3.numModes):
            #print("deltadk: ",np.sum(qm3.TPred[k]),np.sum(qm3_base.TPred[k]))
            self.Ck1[k] = jnp.sum(qm3_base.TPred[k]) #baseline count people
            self.Ck2[k] = jnp.sum(qm3.TPred[k]) #scenario count people
            self.CkDiff[k] = self.Ck2[k] - self.Ck1[k] #difference in people by mode
        
        #compute distance travelled on modes
        self.Lk1 = self.computeL_k(qm3_base.TPred,dijKM) #baseline distance travelled
        self.Lk2 = self.computeL_k(qm3.TPred,dijKM) #scenario distance travelled
        self.deltaLk = [self.Lk2[k]-self.Lk1[k] for k in range(0,qm3.numModes)] #difference in distance by mode

        #compute scenario link statistics - measures changes made by the network changes directly
        self.scenarioLinkDepth_k, self.scenarioLinkKM_k, self.scenarioLinkSavedSecs_k \
            = self.computeScenarioLinkStatistics(networkChanges, qm3_base.Cij, dijKM)
        
        #compute LBar, geographic spread of scenario
        self.LBar_k = self.computeLBar(networkChanges, dijKM)

        #compute scenario network statistics - measures number of faster trips and saved time (secondary changes as a result of APSP)
        self.nMinus_k, self.savedSecs_k = self.computeNetworkStatistics(qm3_base.Cij, qm3.Cij)

################################################################################

    """
    computeZones
    Computes statistics per zone, rather than overall total statistics for a scenario
    """
    def computeZones(self,qm3_base: SingleOrigin,qm3: SingleOrigin, dijKM:np.matrix, networkChanges: list):
        (M, N) = np.shape(qm3.TObs[0])

         #count people on modes
        self.Cik1 = np.zeros((qm3.numModes,N),dtype=float)
        self.Cik2 = np.zeros((qm3.numModes,N),dtype=float)
        self.CikDiff = np.zeros((qm3.numModes,N),dtype=float)
        self.Cjk1 = np.zeros((qm3.numModes,N),dtype=float)
        self.Cjk2 = np.zeros((qm3.numModes,N),dtype=float)
        self.CjkDiff = np.zeros((qm3.numModes,N),dtype=float)
        
        #calculations
        for k in range(0,qm3.numModes):
            self.Cik1[k,:] = jnp.sum(qm3_base.TPred[k],axis=1) #baseline count people
            self.Cik2[k,:] = jnp.sum(qm3.TPred[k],axis=1) #scenario count people
            self.CikDiff[k,:] = self.Cik2[k,:] - self.Cik1[k,:] #difference in people by mode


################################################################################

    def writeStatisticsFile(self,filename,df_zonecodes):
        (M,N) = np.shape(self.Cik1)
        with filename.open('w') as f:
            #header line
            f.write('zonei,zonecode,'
                    +'Cik1_road,Cik1_bus,Cik1_rail,'
                    +'Cik2_road,Cik2_bus,Cik2_rail,'
                    +'CikDiff_road,CikDiff_bus,CikDiff_rail,'
                    +'Cjk1_road,Cjk1_bus,Cjk1_rail,'
                    +'Cjk2_road,Cjk2_bus,Cjk2_rail,'
                    +'CjkDiff_road,CjkDiff_bus,CjkDiff_rail\n')
            for i in range(0,N):
                row = df_zonecodes[df_zonecodes['zonei'] == i]
                zonecode=row['areakey'].values[0]
                f.write(str(i)+','+zonecode+',')
                f.write(f'{self.Cik1[0,i]},{self.Cik1[1,i]},{self.Cik1[2,i]},')
                f.write(f'{self.Cik2[0,i]},{self.Cik2[1,i]},{self.Cik2[2,i]},')
                f.write(f'{self.CikDiff[0,i]},{self.CikDiff[1,i]},{self.CikDiff[2,i]},')
                f.write(f'{self.Cjk1[0,i]},{self.Cjk1[1,i]},{self.Cjk1[2,i]},')
                f.write(f'{self.Cjk2[0,i]},{self.Cjk2[1,i]},{self.Cjk2[2,i]},')
                f.write(f'{self.CjkDiff[0,i]},{self.CjkDiff[1,i]},{self.CjkDiff[2,i]}')
                f.write('\n')
            #end for
    ###

################################################################################


#end class