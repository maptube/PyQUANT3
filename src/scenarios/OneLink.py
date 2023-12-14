"""
onelink.py
Scenario generator for exhaustively building single link scenarios.
"""

from models.DirectNetworkChange import DirectNetworkChange

class OneLinkLimitR:

    """
    Constructor
    @param maxRadiusKM maximum length of links which are allowed e.g. 10KM
    @param numZones number of zones in the model, which limits the i and j numbers generated
    @param mode the mode number which goes into the DirectNetworkChange e.g. 0=road, 1=bus, 2=rail
    @param LijKM the distance matrix (crowfly vertex km) matching the mode param above, used to limit distances to maxRadiusKM
    """
    def __init__(self,maxRadiusKM,numZones,mode,LijKM):
        self.maxRadiusKM = maxRadiusKM
        self.N = numZones
        self.mode = mode
        self.LijKM = LijKM
        self.i = 0 #origin
        self.j = 0 #destination

################################################################################

    """
    next
    Gets the next scenario in the sequence.
    @returns a DirectNetworkChange with a single link satisfying the maxRadiusKM constraint.
    NOTE:
    it returns the empty list if there are no scenarios left that are possible.
    NOTE 2:
    it always returns a DirectNetworkChange with -1 as the link time, which you have to fill in later
    """
    def next(self):
        result = []
        finished=False
        while not finished:
            #this is to increment j and i to the next valid scenario
            self.j+=1
            if (self.j>=self.N):
                self.j=0
                self.i+=1
                if self.i>=self.N:
                    finished=True #no more scenarios - we're finished
            #end if
            
            #now the check on maxRadiusKM and either break and exit or continue with j and i
            if not finished and self.LijKM[self.i,self.j]<=self.maxRadiusKM:
                result = [ DirectNetworkChange(self.mode,self.i,self.j,-1) ]
                finished=True #and we drop out of the while loop
            #end if
        #end while

        return result

################################################################################
