"""
ImpactStatistics.py
Compute impact statistics for a set of matrices from a scenario run
"""

import numpy as np

class ImpactStatistics:
    def __init__(self):
        self.deltaDk = []
        self.deltaLk = []
    
    ################################################################################

    """
        compute
        Computes impact statics given a baseline model and a scenario changes model
        @param qm3_base baseline quant 3 model
        @param qm3 scenario quant 3 model
    """
    def compute(self,qm3_base,qm3):
        (M, N) = np.shape(qm3.TObs[0])

        self.deltaDk = [0 for k in range(0,qm3.numModes)]
        self.deltaLk = [0 for k in range(0,qm3.numModes)]

        for k in range(0,qm3.numModes):
            print("deltadk: ",np.sum(qm3.TPred[k]),np.sum(qm3_base.TPred[k]))
            self.deltaDk[k] = np.sum(qm3.TPred[k]) - np.sum(qm3_base.TPred[k])

    ################################################################################


#end class