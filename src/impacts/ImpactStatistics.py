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
    compute
    Computes impact statics given a baseline model and a scenario changes model
    @param qm3_base baseline quant 3 model
    @param qm3 scenario quant 3 model
    @param dijKM vertex KM distance file, 3 modes
    """
    def compute(self,qm3_base,qm3, dijKM):

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

################################################################################


#end class