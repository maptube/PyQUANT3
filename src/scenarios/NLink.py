"""
NLink.py
Scenario generator for exhaustively building 'N' link scenarios.
N=2 gives you two link scenarios, 3=three link etc.
NOTE: don't confuse this N (=LinkN) with the other N, which is number of zones (=8436)
"""

import random
from models.DirectNetworkChange import DirectNetworkChange

class NLinkLimitR:

    """
    Constructor
    @param linkN number of links in the scenario
    @param maxRadiusKM maximum length of links which are allowed e.g. 10KM
    @param numZones number of zones in the model, which limits the i and j numbers generated
    @param mode the mode number which goes into the DirectNetworkChange e.g. 0=road, 1=bus, 2=rail
    @param LijKM the distance matrix (crowfly vertex km) matching the mode param above, used to limit distances to maxRadiusKM
    """
    def __init__(self,linkN,maxRadiusKM,numZones,mode,LijKM):
        self.linkN = linkN
        self.maxRadiusKM = maxRadiusKM
        self.N = numZones
        self.mode = mode
        self.LijKM = LijKM
        self.i = 0 #origin
        self.j = -1 #destination

################################################################################

    """
    next
    Gets the next scenario in the sequence, which is comprised of N links.
    @returns a DirectNetworkChange with a single link satisfying the maxRadiusKM constraint.
    NOTE:
    it returns the empty list if there are no scenarios left that are possible.
    NOTE 2:
    it always returns a DirectNetworkChange with -1 as the link time, which you have to fill in later
    """
    def next(self):
        result = []
        backlink=-1 #used to make sure we don't backtrace - it's the previous origin link

        #pick a first origin - can be any of the N zones
        self.i = random.randint(0,self.N-1) #note, they're both inclusive limits
        for n in range(0,self.linkN):
            #select all destinations within maxRadiusKM of i
            list = []
            for nj in range(0,self.N):
                #make sure we don't back track and pick a possible within maxRadiusKM
                if nj!=backlink and self.LijKM[self.i,nj]<=self.maxRadiusKM:
                    list.append(nj)
            if (len(list)>0):
                #pick a random one
                self.j = random.choice(list)
                #log it
                result.append( DirectNetworkChange(self.mode,self.i,self.j,-1) )
                #and then i=j to move the link along the chain
                backline=self.i # set back link to the one before the next origin so we don't backtrack
                self.i=self.j
            #endif
        #end for

        return result

################################################################################
