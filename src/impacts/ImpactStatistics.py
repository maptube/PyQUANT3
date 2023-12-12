"""
ImpactStatistics.py
Compute impact statistics for a set of matrices from a scenario run
"""

import numpy as np

class ImpactStatistics:
    def __init__(self):
        self.Ck1 = [] #baseline population count (people) by mode
        self.Ck2 = [] #scenario population count (people) by mode
        self.CkDiff = [] #Ck2-Ck1 scenario minus baseline
        #self.deltaDk = []
        self.Lk1 = [] #baseline distance (KM) travelled on each mode
        self.Lk2 = [] #scenario distance (KM) travelled on each mode
        self.deltaLk = [] #Lk2=Lk1 difference in distance travelled by mode

        #scenario link network changes - these are physical changes made by the scenario
        self.scenarioLinkDepth_k = [] #this is how many changed links there are on each mode
        self.scenarioLinkKM_k = [] #distances of added links by mode (KM)
        self.scenarioLinkSavedSecs_k = [] #times saved by new added linsk by mode (secs)

        #scenario network changes - these are results of the APSP algorithm on network structure
        self.nMinus_k = []
        self.savedSecs_k = []
    
################################################################################

    """
    Compute total distance travelled on each mode
    @param Tij FMatrix[] trips matrix
    @param dijKM FMatrix[] distance matrix
    @returns double [] total distance travelled by people
    """
    def computeL_k(self, Tij, dijKM):
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
            #faster
            Lk[k]=np.sum(Tij[k] * dijKM[k])
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
    computeNetworkStatistics
    This is derived network statistics (secondary) as a result of the primary
    changes from the network scenario links that are added.
    nMinus - number of (geometric) trips that are now quicker
    savedSecs - number of seconds saved by the new shortest path geometry
    @param Cij1 baseline shortest paths (minutes) matrix by mode
    @param Cij2 scenario shortest paths (minutes) matrix by mode
    @returns nMinus, savedSecs
    """
    def computeNetworkStatistics(self,Cij1,Cij2):
        NumModes = len(Cij1)
        nMinus_k = [0.0 for k in range(0,NumModes)]
        savedSecs_k = [0.0 for k in range(0,NumModes)]
        for k in range(0,NumModes):
            nMinus_k[k]=np.count_nonzero(Cij2[k] < Cij1[k])
            diff = Cij2[k]-Cij1[k]
            diff = np.where(diff>0,diff,0) #filter out any negative values
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
    def compute(self,qm3_base,qm3, dijKM, networkChanges):

        (M, N) = np.shape(qm3.TObs[0])

        #count people on modes
        self.Ck1 = [0 for k in range(0,qm3.numModes)]
        self.Ck2 = [0 for k in range(0,qm3.numModes)]
        self.CkDiff = [0 for k in range(0,qm3.numModes)]
        for k in range(0,qm3.numModes):
            #print("deltadk: ",np.sum(qm3.TPred[k]),np.sum(qm3_base.TPred[k]))
            self.Ck1[k] = np.sum(qm3_base.TPred[k]) #baseline count people
            self.Ck2[k] = np.sum(qm3.TPred[k]) #scenario count people
            self.CkDiff[k] = self.Ck2[k] - self.Ck1[k] #difference in people by mode
        
        #compute distance travelled on modes
        self.Lk1 = self.computeL_k(qm3_base.TPred,dijKM) #baseline distance travelled
        self.Lk2 = self.computeL_k(qm3.TPred,dijKM) #scenario distance travelled
        self.deltaLk = [self.Lk2[k]-self.Lk1[k] for k in range(0,qm3.numModes)] #difference in distance by mode

        #compute scenario link statistics - measures changes made by the network changes directly
        self.scenarioLinkDepth_k, self.scenarioLinkKM_k, self.scenarioLinkSavedSecs_k \
            = self.computeScenarioLinkStatistics(networkChanges, qm3_base.Cij, dijKM)

        #compute scenario network statistics - measures number of faster trips and saved time (secondary changes as a result of APSP)
        self.nMinus_k, self.savedSecs_k = self.computeNetworkStatistics(qm3_base.Cij, qm3.Cij)

################################################################################


#end class