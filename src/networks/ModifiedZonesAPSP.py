"""
ModifiedZonesAPSP
Algorithm for modified all pairs shortest paths on a fully connected zone to zone costs matrix i.e. the QUANT "dis" distance and cost matrices.

TODO: the walk of N x N matrix is terrible for Python performance - how to do this?
"""

from numba import jit
import numpy as np

class ModifiedZonesAPSP:
    
    """
    Modified all pairs shortest paths on a fully connected zone to zone costs matrix i.e. the QUANT "dis" distance and cost matrices.
    NOTE: the link is directional, so call it with A=>B and B=>A if you want both.
    TODO: why isn't this parallel?
    @param name="dis" The original costs matrix as input, which is edited in place to return the modified matrix with the new shorter paths
    @param name="Origin" The origin zone, assumed to be directional, so link is O=>D only
    @param name="Destination" The destination zone, assumed to be directional, so link is O=>D only
    @returns The number of shorter paths found and The sum of the minutes saved on all the shorter paths found.
    """
    #@classmethod @staticmethod
    @staticmethod
    @jit(nopython=True)
    def computeModAPSP(dis, Origin, Destination, NewCost):
        #OK, so we have a fully connected costs matrix containing the shortest paths to start with.
        #Then we add a new link Origini to Destj with a new (lower cost). If it's a higher cost, then you might as well just throw that out and return.
        #
        #dis[Origin, Destination] = NewCost; //just in case they haven't already changed it
        #
        #need to check every path going through this new link for a shorter one
        #so, Zonei->Origin+NewCost+Dest->Zonej, for all zones i and j, any link that is reduced for i->j is replaced with the new one via OD

        #yes, it's another walk of NxN, but maybe we can shortcut?
        (M, N) = np.shape(dis)
        count = 0
        totalMinsSaved = 0
        for i in range(0,N):
            io = dis[i, Origin]
            #if (i == Origin) continue; NO! don't do this!
            for j in range(0,N):
                #if (j == Destination) continue; NO! don't do this!
                dj = dis[Destination, j]
                #if io+dj<dist continue as you can't win
                #test here
                dist = io + NewCost + dj
                if dist<dis[i,j]:
                    totalMinsSaved += dis[i, j] - dist; #yes, saved=current-new shorter
                    dis[i, j] = dist
                    count+=1
                #end if
            #end for j
        #end for i
        return count, totalMinsSaved #and the dis matrix that we changed
    
    ################################################################################
