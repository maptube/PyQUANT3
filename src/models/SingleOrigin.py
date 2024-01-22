from numba import jit
import numpy as np
from math import exp, fabs
from sys import float_info
import time
import copy

from networks.ModifiedZonesAPSP import ModifiedZonesAPSP


"""
Single destination constrained gravity model
"""
class SingleOrigin:
    #self.NumModes=3 #const???
    
###############################################################################

    def __init__(self):
        self.numModes=3
        self.TObs=[] #Data input to model list of NDArray
        self.Cij=[] #cost matrix for zones in TObs

        self.isUsingConstraints = False
        self.constraints = [] #1 or 0 to indicate constraints for zones matching TObs - this applies to all modes

        self.TPred=[] #this is the output
        self.B=[] #this is the constraints output vector - this applies to all modes
        self.Beta=[] #Beta values for three modes - this is also output

        self.CBarObs=[] #average trip time minutes (mode) observed
        self.CBarPred=[] #average trip time minutes (mode) predicted

    """
    calculateCBar
    Mean trips calculation
    @param name="Tij" NDArray
    @param name="cij" NDArray
    @returns float
    """
    #@jit(nopython=True)
    #@staticmethod
    def calculateCBar(self,Tij,cij):
        #(M, N) = np.shape(Tij)
        #CNumerator = 0.0
        #CDenominator = 0.0
        #for i in range(0,N):
        #    for j in range(0,N):
        #        CNumerator += Tij[i, j] * cij[i, j]
        #        CDenominator += Tij[i, j]
        #CBar = CNumerator / CDenominator
        #print("CBar=",CBar)
        #faster
        CNumerator2 = np.sum(Tij*cij)
        CDenominator2 = np.sum(Tij)
        CBar2=CNumerator2/CDenominator2
        #print("CBar2=",CBar2)
        #cupy example
        #CNumerator2 = cp.sum(Tij*cij)
        #CDenominator2 = cp.sum(Tij)
        #CBar2 = CNumerator2/CDenominator2

        return CBar2

###############################################################################

    """
    deepcopy
    Return a deep copy version of self
    """
    #@jit(nopython=True)
    def deepcopy(self):
        qm3 = SingleOrigin()
        qm3.numModes = self.numModes
        qm3.isUsingConstraints = self.isUsingConstraints
        qm3.constraints = copy.deepcopy(self.constraints)
        qm3.Beta = copy.deepcopy(self.Beta)
        #qm3.Beta = [ self.Beta[k] for k in range(0,self.numModes) ]
        #b = np.array(a, copy=True) or b = np.copy(a) ???
        #qm3.TObs = copy.deepcopy(self.TObs)
        #qm3.TObs = [ np.array(self.TObs[k],copy=True) for k in range(0,self.numModes) ]
        qm3.TObs = [ np.copy(self.TObs[k]) for k in range(0,self.numModes) ]
        #qm3.TObs = [deepcopyarray(self.TObs[k]) for k in range(0,self.numModes)]
        #qm3.TPred = copy.deepcopy(self.TPred)
        #qm3.TPred = [ np.array(self.TPred[k],copy=True) for k in range(0,self.numModes) ]
        qm3.TPred = [ np.copy(self.TPred[k]) for k in range(0,self.numModes) ]
        #qm3.TPred = [ deepcopyarray(self.TPred[k]) for k in range(0,self.numModes) ]
        #qm3.Cij = copy.deepcopy(self.Cij)
        #qm3.Cij = [ np.array(self.Cij[k],copy=True) for k in range(0,self.numModes) ]
        qm3.Cij = [ np.copy(self.Cij[k]) for k in range(0,self.numModes) ]
        #qm3.Cij = [ deepcopyarray(self.Cij[k]) for k in range(0,self.numModes) ]
        
        #qm3.TObs = [np.array(1) for k in range(0,self.numModes)]
        #qm3.TPred = [np.array(1) for k in range(0,self.numModes)]
        #qm3.Cij = [np.array(1) for k in range(0,self.numModes)]
        #for k in range(0,3):
        #    qm3.TObs[k] = np.copy(self.TObs[k])
        #    qm3.TPred[k] = np.copy(self.TPred[k])
        #    qm3.Cij[k] = np.copy(self.Cij[k])
        
        qm3.B = copy.deepcopy(self.B)
        return qm3


###############################################################################

    """
    Calculate Oi for a trips matrix.
    Two methods are presented here, one which is simple and very slow and one
    which uses python vector maths and is much faster. Once 2 is proven equal
    to 1, then it can be used exclusively. This function is mainly used for
    testing with the TensorFlow and other implementations.
    """
    #@jit(nopython=True)
    @staticmethod
    def calculateOi(Tij: np.matrix) -> np.array:
        (M, N) = np.shape(Tij)
        #OiObs
        Oi = np.zeros(N)
        #Method 1 - slow, but simplest implementation for testing with
        #for i in range(0,N):
        #    sum = 0.0
        #    for j in range(0,N):
        #        sum += Tij[i, j]
        #    Oi[i] = sum
        #Method 2 - MUCH FASTER! But check that it is identical to method 1
        Oi=Tij.sum(axis=1)
        return Oi

###############################################################################

    """
    Calculate Dj for a trips matrix.
    Two methods are presented here, one which is simple and very slow and one
    which uses python vector maths and is much faster. Once 2 is proven equal
    to 1, then it can be used exclusively. This function is mainly used for
    testing with the TensorFlow and other implementations.
    """
    #@jit(nopython=True)
    @staticmethod
    def calculateDj(Tij: np.matrix) -> np.array:
        (M, N) = np.shape(Tij)
        #DjObs
        Dj = np.zeros(N)
        #Method 1 - slow, but simplest implementation for testing with
        #for j in range(0,N):
        #    sum = 0.0
        #    for i in range(0,N):
        #        sum += Tij[i, j]
        #    Dj[j] = sum
        #Method 2 - MUCH FASTER! But check that it is identical to method 1
        Dj=Tij.sum(axis=0)
        return Dj

###############################################################################

    """
    fastComputePredicted
    Computes the predicted matrices based on betas and TObs having previously
    been set from a calibration run. Used in scenarios for the baseline
    matrices.
    PRE: betas, TObs and Cij all set
    POST: TPred all set
    """
    #@jit(nopython=True)
    def fastComputePredicted(self):
        (M, N) = np.shape(self.TObs[0])
        
        #Oi obs
        ksum=np.array([np.zeros(N),np.zeros(N),np.zeros(N)])
        for k in range(0,self.numModes):
            ksum[k]=self.TObs[k].sum(axis=1)
        OiObs = ksum.sum(axis=0)

        #Dj obs
        ksum=np.array([np.zeros(N),np.zeros(N),np.zeros(N)])
        for k in range(0,self.numModes):
            ksum[k]=self.TObs[k].sum(axis=0)
        DjObs = ksum.sum(axis=0)

        #constraints initialisation
        B = [1.0 for i in range(0,N)] #hack
        Z = [0.0 for i in range(0,N)]
        for j in range(0,N):
            Z[j] = 1.7976931348623157e+308 #float_info.max
            if self.isUsingConstraints:
                if self.constraints[j] >= 1.0: #constraints array taking place of Gj (Green Belt) in documentation
                    #Gj=1 means 0.8 of MSOA land is green belt, so can't be built on
                    #set constraint to original Dj
                    Z[j] = DjObs[j]
        #end of constraints initialisation - have now set B[] and Z[] based on IsUsingConstraints, Constraints[] and DObs[]

        #pre-calculate exp(-Beta[k]*self.Cij[k]) for speed
        expBetaCij = [np.exp(-self.Beta[k]*self.Cij[k]) for k in range(0,self.numModes)]
        self.TPred = [np.zeros(N*N).reshape(N, N) for k in range(0,self.numModes) ]
        for k in range(0,self.numModes): #mode loop
            for i in range(0,N):
                denom2=0.0
                for kk in range(0,self.numModes):
                    denom2+=np.sum(DjObs*expBetaCij[kk][i,:])

                #numerator calculation for this mode (k)
                Tijk2=OiObs[i]*(B*DjObs*expBetaCij[k][i]/denom2)
                self.TPred[k][i,:]=Tijk2 #put answer slice back in return array 
            #end for i
        #end for k
    #end def

###############################################################################

    """
    run
    Calibrate model - yes, I know, but it was always called "run" for some reason
    and it's stuck. It's really calibrate, but you have to run the model over
    and over again while you tune the betas, so I guess it's run.
    @returns nothing
    """
    @jit(nopython=True)
    def run(self):
        (M, N) = np.shape(self.TObs[0])
        
        #set up Beta for modes 0, 1 and 2 to 1.0f
        self.Beta = [1.0 for k in range(0,self.numModes)]

        #work out Dobs and Tobs from rows and columns of TObs matrix
        #These don't ever change so they need to be outside the convergence loop
        DjObs = np.zeros(N) #np.arange(N)
        OiObs = np.zeros(N) #np.arange(N)
        #sum=0.0

        #OiObs
        #for i in range(0,N):
        #    sum = 0.0
        #    for j in range(0,N):
        #        for k in range(0, self.numModes):
        #            sum += self.TObs[k][i, j]
        #    OiObs[i] = sum
        #MUCH FASTER!
        ksum=np.array([np.zeros(N),np.zeros(N),np.zeros(N)])
        for k in range(0,self.numModes):
            ksum[k]=self.TObs[k].sum(axis=1)
        OiObs = ksum.sum(axis=0)
        #print("check 1: ",OiObs[0],ksum[0][0]+ksum[1][0]+ksum[2][0])
        #print("check 1: ",OiObs2[0])
        #for i in range(0,N):
        #    print(OiObs[i],OiObs2[i])

        #DjObs
        #for j in range(0,N):
        #    sum = 0.0
        #    for i in range(0,N):
        #        for k in range(0,self.numModes):
        #            sum += self.TObs[k][i, j]
        #    DjObs[j] = sum
        #MUCH FASTER!
        ksum=np.array([np.zeros(N),np.zeros(N),np.zeros(N)])
        for k in range(0,self.numModes):
            ksum[k]=self.TObs[k].sum(axis=0)
        DjObs = ksum.sum(axis=0)
        #for i in range(0,N):
        #    print(DjObs[i],DjObs2[i])

        print("OiObs and DjObs calculated")

        #constraints initialisation
        B = [1.0 for i in range(0,N)] #hack
        Z = [0.0 for i in range(0,N)]
        for j in range(0,N):
            Z[j] = float_info.max
            if self.isUsingConstraints:
                if self.constraints[j] >= 1.0: #constraints array taking place of Gj (Green Belt) in documentation
                    #Gj=1 means 0.8 of MSOA land is green belt, so can't be built on
                    #set constraint to original Dj
                    Z[j] = DjObs[j]
        #end of constraints initialisation - have now set B[] and Z[] based on IsUsingConstraints, Constraints[] and DObs[]


        Tij = [np.zeros(N*N).reshape(N, N) for k in range(0,self.numModes) ] #array of matrix over each mode(k) - need declaration outside loop
        converged = False
        while not converged:      
            constraintsMet = False
            while not constraintsMet:
                #residential constraints
                constraintsMet = True #unless violated one or more times below
                failedConstraintsCount = 0

                #model run
                #Tij = [np.zeros(N*N).reshape(N, N) for k in range(0,self.numModes) ]
                #pre-calculate exp(-Beta[k]*self.Cij[k]) for speed
                expBetaCij = [np.exp(-self.Beta[k]*self.Cij[k]) for k in range(0,self.numModes)]
                for k in range(0,self.numModes): #mode loop
                    print("Running model for mode ",k)
                    #Tij[k] = np.zeros(N*N).reshape(N, N)

                    for i in range(0,N):
                        #denominator calculation which is sum of all modes
                        #denom = 0.0  #double
                        #for kk in range(0,self.numModes): #second mode loop
                        #    for j in range(0,N):
                        #        denom += DjObs[j] * exp(-Beta[kk] * self.Cij[kk][i, j])
                        #    #end for j
                        ##end for kk
                        #print("denom=",denom)
                        #faster...?
                        denom2=0.0
                        for kk in range(0,self.numModes):
                            #expBetaCij=np.exp(-Beta[kk]*self.Cij[kk])
                            #print("expBetaCij=",expBetaCij)
                            denom2+=np.sum(DjObs*expBetaCij[kk][i,:])
                        #print("denom2=",denom2)

                        #numerator calculation for this mode (k)
                        #for j in range(0,N):
                        #    Tij[k][i, j] = B[j] * OiObs[i] * DjObs[j] * exp(-Beta[k] * self.Cij[k][i, j]) / denom
                        #print("Tijk[0,0]=",Tij[k][i,0])
                        #faster
                        Tijk2=OiObs[i]*(B*DjObs*expBetaCij[k][i]/denom2)
                        Tij[k][i,:]=Tijk2 #put answer slice back in return array 
                        #print("Tijk2[0,0]=",Tijk2[0])
                    #end for i
                #end for k

                #constraints check
                if self.isUsingConstraints:
                    print("Constraints test")

                    for j in range(0,N):
                        Dj = 0.0
                        for i in range(0,N): Dj += Tij[0][i,j]+Tij[1][i,j]+Tij[2][i,j]
                        if self.constraints[j] >= 1.0: #Constraints is taking the place of Gj in the documentation
                            if (Dj - Z[j]) >= 0.5: #was >1.0
                                B[j] = B[j] * Z[j] / Dj
                                constraintsMet = False
                                failedConstraintsCount+=1
                                print("Constraints violated on " + failedConstraintsCount + " MSOA zones")
                                print("Dj=", Dj, " Zj=", Z[j], " Bj=", B[j])
                            #end if (Dj-Z[j])>=0.5
                        #end if Constraints[j]>=1.0
                    #end for j
                         
                    #copy B2 into B ready for the next round
                    #for (int j = 0; j < N; j++) B[j] = B2[j];
                #end if self.isUsingConstraints
                print("FailedConstraintsCount=", failedConstraintsCount)

                #Instrumentation block
                #for (int k = 0; k < NumModes; k++)
                #    InstrumentSetVariable("Beta" + k, Beta[k]);
                #InstrumentSetVariable("delta", FailedConstraintsCount); //not technically delta, but I want to see it for testing
                #InstrumentTimeInterval();
                #end of instrumentation block

            #end while not ConstraintsMet

            #calculate mean predicted trips and mean observed trips (this is CBar)
            self.CBarPred = [0.0 for k in range(0,self.numModes)]
            self.CBarObs = [0.0 for k in range(0,self.numModes)]
            delta = [0.0 for k in range(0,self.numModes)]
            for k in range(0,self.numModes):
                self.CBarPred[k] = self.calculateCBar(Tij[k], self.Cij[k])
                self.CBarObs[k] = self.calculateCBar(self.TObs[k], self.Cij[k])
                print("Mode "+str(k)+" CBar Pred="+str(self.CBarPred[k])+" CBar Obs="+str(self.CBarObs[k]))
                delta[k] = fabs(self.CBarPred[k] - self.CBarObs[k]) #the aim is to minimise delta[0]+delta[1]+...
            #end for k

            #delta check on all betas (Beta0, Beta1, Beta2) stopping condition for convergence
            #double gradient descent search on Beta0 and Beta1 and Beta2
            converged = True
            for k in range(0,self.numModes):
                if delta[k] / self.CBarObs[k] > 0.001:
                    self.Beta[k] = self.Beta[k] * self.CBarPred[k] / self.CBarObs[k]
                    converged = False
            #end for k
            #Debug block
            for k in range(0,self.numModes):
                print("Beta", k, "=", self.Beta[k])
                print("delta", k, "=", delta[k])
            #end for k
            #print("delta", delta[0], delta[1], delta[2]) #should be a k loop
            #end of debug block
        #end while not Converged

        #Set the output, TPred[]
        self.TPred = []
        for k in range(0,self.numModes):
            self.TPred.append(Tij[k])

        #debugging:
        #for (int i = 0; i < N; i++)
        #    System.Diagnostics.Debug.WriteLine("Quant3Model::Run::ConstraintsB," + i + "," + B[i]);

###############################################################################

    """
    RunWithChanges
    NOTE: this was copied directly from the Quant 1 model
    Run the quant model with different values for the Oi and Dj zones.
    PRE: needs TObs, cij and beta
    TODO: need to instrument this
    TODO: writes out one file, which is the sum of the three predicted matrices produced
    @param name="OiDjHash" Hashmap of zonei index and Oi, Dj values for that area. A value of -1 for Oi or Dj means no change.
    @param name="NetworkChanges" List<DirectNetworkChange> Changes to network based on MOSA to MSOA direct changes to the dis matrix. The
    zone ids are used to identify rows and columns. Time is in SECONDS. The array is k=0 (road), k=1 (bus), k=2 (rail).
    NOTE: you can pass in null for no changes. At the moment, these are directional, so zone 0->1 DOES NOT affect zone 1->0.
    @param name="hasConstraints" Run with random values added to the Dj values.
    """
    def runWithChanges(self, OiDjHash, NetworkChanges, hasConstraints):

        (M, N) = np.shape(self.TObs[0])

        #DjObs = [0.0 for i in range(0,N)]
        #OiObs = [0.0 for i in range(0,N)]
        #Sum=0.0

        #OiObs
        #for i in range(0,N):
        #    sum = 0.0
        #    for j in range(0,N):
        #        for k in range(0,self.numModes): sum += self.TObs[k][i, j]
        #    #end for j
        #    OiObs[i] = sum
        #end for i
        #MUCH FASTER!
        ksum=np.array([np.zeros(N),np.zeros(N),np.zeros(N)])
        for k in range(0,self.numModes):
            ksum[k]=self.TObs[k].sum(axis=1)
        OiObs = ksum.sum(axis=0)

        #DjObs
        #for j in range(0,N):
        #    sum = 0.0
        #    for i in range(0,N):
        #        for k in range(0,self.numModes): sum += self.TObs[k][i, j]
        #    #end for i
        #    DjObs[j] = sum
        #end for j
        #MUCH FASTER!
        ksum=np.array([np.zeros(N),np.zeros(N),np.zeros(N)])
        for k in range(0,self.numModes):
            ksum[k]=self.TObs[k].sum(axis=0)
        DjObs = ksum.sum(axis=0)

        #this is a complete hack - generate a TPred matrix that we can get Dj constraints from
        TPredCons = [np.arange(N*N,dtype=np.float64).reshape(N, N) for k in range(0,self.numModes) ]
        #pre-calculate exp(-Beta[k]*self.Cij[k]) for speed
        expBetaCij = [np.exp(-self.Beta[k]*self.Cij[k]) for k in range(0,self.numModes)]
        for k in range(0,self.numModes): #mode loop
            #TPredCons[k] = np.arange(N*N).reshape(N,N)

            for i in range(0,N):
                #denominator calculation which is sum of all modes
                #denom = 0.0
                #for kk in range(0,self.numModes): #second mode loop
                #    for j in range(0,N):
                #        denom += DjObs[j] * exp(-self.Beta[kk] * self.Cij[kk][i, j])
                #    #end for j
                #end for kk
                #faster...?
                denom2=0.0
                for kk in range(0,self.numModes):
                    #expBetaCij=np.exp(-Beta[kk]*self.Cij[kk])
                    #print("expBetaCij=",expBetaCij)
                    denom2+=np.sum(DjObs*expBetaCij[kk][i,:])

                #numerator calculation for this mode (k)
                #for j in range(0,N):
                #    TPredCons[k][i, j] = self.B[j] * OiObs[i] * DjObs[j] * exp(-self.Beta[k] * self.Cij[k][i, j]) / denom
                #end for j
                #faster
                Tijk2=OiObs[i]*(self.B*DjObs*expBetaCij[k][i]/denom2)
                TPredCons[k][i,:]=Tijk2 #put answer slice back in return array
            #end for i
        #end for k
        #now the DjCons - you could just set Zj here?
        DjCons = [0.0 for j in range(0,N)]
        #for j in range(0,N):
        #    sum = 0.0
        #    for i in range(0,N):
        #        for k in range(0,self.numModes): sum += TPredCons[k][i, j]
        #    #end for i
        #    DjCons[j] = sum
        #end for j
        #MUCH FASTER!
        ksum=np.array([np.zeros(N),np.zeros(N),np.zeros(N)])
        for k in range(0,self.numModes):
            ksum[k]=TPredCons[k].sum(axis=0)
        DjCons = ksum.sum(axis=0)
            
        #
        #
        #TODO: Question - do the constraints take place before or after the Oi Dj changes? If before, then it's impossible to increase jobs in greenbelt zones. If after, then changes override the green belt.

        #constraints initialisation - this is the same as the calibration, except that the B[j] values are initially taken from the calibration, while Z[j] is initialised from Dj[j] as before.
        Z = [0.0 for j in range(0,N)]
        for j in range(0,N):
            Z[j] = 1.7976931348623157e+308 #float_info.max
            if self.isUsingConstraints:
                if self.constraints[j] >= 1.0: #constraints array taking place of Gj (Green Belt) in documentation
                    #Gj=1 means a high enough percentage of MSOA land is green belt, so can't be built on
                    #set constraint to original Dj
                    #Z[j] = DjObs[j];
                    Z[j] = DjCons[j]
                #end if constraints[j]
            #end if self.isUsingConstraints
        #end for j
        #end of constraints initialisation - have now set B[] and Z[] based on IsUsingConstraints, Constraints[] and DObs[]

        #apply changes here from the hashmap
        for key in OiDjHash:
            i = int(key)
            value = OiDjHash[i]
            if value[0] >= 0: OiObs[i] = value[0]
            if value[1] >= 0: DjObs[i] = value[1]
        #end for key

        #apply network changes - these are directly made to the dis matrices
        if NetworkChanges: #test against none type
            #InstrumentStatusText = "Making network changes";
            count = 0
            countMode = [ 0, 0, 0 ]
            for dnc in NetworkChanges:
                self.Cij[dnc.mode][dnc.originZonei, dnc.destinationZonei] = dnc.absoluteTimeSecs / 60.0 #seconds to minutes (NOTE: this is done in ComputeModAPSP anyway)
                #and add the reverse link
                self.Cij[dnc.mode][dnc.destinationZonei, dnc.originZonei] = dnc.absoluteTimeSecs / 60.0 #seconds to minutes
                #compute secondary links - both ways around
                #int linkCount1, linkCount2;
                #float totalMinsSaved1, totalMinsSaved2;
                start_time = time.process_time()
                linkCount1, totalMinsSaved1 = ModifiedZonesAPSP.computeModAPSP(self.Cij[dnc.mode], dnc.originZonei, dnc.destinationZonei, dnc.absoluteTimeSecs / 60.0)
                end_time1 = time.process_time()
                linkCount2, totalMinsSaved2 = ModifiedZonesAPSP.computeModAPSP(self.Cij[dnc.mode], dnc.destinationZonei, dnc.originZonei, dnc.absoluteTimeSecs / 60.0)
                end_time2 = time.process_time()
                print("ModifiedZonesAPSP:: ASPS1="+str(end_time1-start_time)+" APSP2="+str(end_time2-end_time1)+" secs")
                count += linkCount1 + linkCount2
                countMode[dnc.mode] += linkCount1 + linkCount2
                #totalMinsSaved = totalMinsSaved1+totalMinsSaved2; //if you need the minutes saved...
            #endfor
            #write out changed dis matrices if necessary
            #if (countMode[(int)QUANT3Modes.Q3Road] > 0)
            #    dis[(int)QUANT3Modes.Q3Road].DirtySerialise(Path.Combine(rootdir,"dis_road.bin"));

            print("QUANTModel3::RunWithChanges ComputeModAPSP links changed = ",count)
            #now need to recompute exp(-beta * Cij) as Cij has changed!
            expBetaCij = [np.exp(-self.Beta[k]*self.Cij[k]) for k in range(0,self.numModes)]
        #endif network changes!=null


        constraintsMet = False
        while not constraintsMet:
            #residential constraints
            constraintsMet = True #unless violated one or more times below
            failedConstraintsCount = 0

            #run 3 model
            print("Run 3 model")
            self.TPred=[]
            for k in range(0,self.numModes): #mode loop
                #self.TPred[k] = np.arange(N*N).reshape(N,N)
                self.TPred.append(np.arange(N*N,dtype=np.float64).reshape(N,N))

                for i in range(0,N):
                    #denominator calculation which is sum of all modes
                    #denom = 0.0
                    #for kk in range(0,self.numModes): #second mode loop
                    #    for j in range(0,N):
                    #        denom += self.B[j]*DjObs[j] * exp(-self.Beta[kk] * self.Cij[kk][i, j])
                    #end for kk
                    #faster...?
                    denom2=0.0
                    for kk in range(0,self.numModes):
                        #expBetaCij=np.exp(-Beta[kk]*self.Cij[kk])
                        #print("expBetaCij=",expBetaCij)
                        denom2+=np.sum(DjObs*expBetaCij[kk][i,:])

                    #numerator calculation for this mode (k)
                    #for j in range(0,N):
                    #    self.TPred[k][i, j] = self.B[j] * OiObs[i] * DjObs[j] * exp(-self.Beta[k] * self.Cij[k][i, j]) / denom
                    #faster
                    Tijk2=OiObs[i]*(self.B*DjObs*expBetaCij[k][i]/denom2)
                    self.TPred[k][i,:]=Tijk2 #put answer slice back in return array
                #end for i
            #end for k

            #constraints check
            if self.isUsingConstraints:
                print("Constraints test")

                for j in range(0,N):
                    Dj = 0.0
                    for i in range(0,N): Dj += self.TPred[0][i, j] + self.TPred[1][i, j] + self.TPred[2][i, j]
                    if self.constraints[j] >= 1.0: #Constraints is taking the place of Gj in the documentation
                        #System.Diagnostics.Debug.WriteLine("Test: " + Dj + ", " + Z[j] + "," + B[j]);
                        if (Dj - Z[j]) >= 0.5: #was >1.0 threshold
                            self.B[j] = self.B[j] * Z[j] / Dj
                            constraintsMet = False
                            failedConstraintsCount+=1
#                                System.Diagnostics.Debug.WriteLine("Constraints violated on " + FailedConstraintsCount + " MSOA zones");
#                                System.Diagnostics.Debug.WriteLine("Dj=" + Dj + " Zj=" + Z[j] + " Bj=" + B[j]);
                        #end if (D[j]-Z[j])>=0.5
                    #end if Constraints[j]>=1.0
                #end for j
            #end if self.isUsingConstraints
        #end while not constraintsMet

        #add all three TPred together
        #TPredAll = np.arange(N*N).reshape(N,N)
        #for i in range(0,N):
        #    for j in range(0,N):
        #        Sum = 0.0
        #        for k in range(0,self.numModes):
        #            Sum += self.TPred[k][i, j]
        #        #end for k
        #        TPredAll[i, j] = Sum
        #    #end for j
        ##end for i
        #faster
        #for k in range(0,self.numModes):
        #    TPredAll += self.TPred[k]
        
            
        #and store the result somewhere
        #TPredAll.DirtySerialise(OutFilename);
        #return TPredAll
    #end def runWithChanges


###############################################################################

    """
    Added to allow timing of the main loop for speed comparison with the TensorFlow code.
    NOTE: this is only single mode and one iteration of the main loop. This is what all the benchmark tests do.
    """
    def benchmarkRun(self,numRuns,Tij,Cij,Beta):
        #run Tij = Ai * Oi * Dj * exp(-Beta * Cij)   where Ai = 1/sumj Dj*exp(-Beta * Cij)
        (M, N) = np.shape(Tij)
        starttime = time.time()
        for r in range(0,numRuns):
            TPred = np.zeros(N*N).reshape(N, N)
            Oi = self.calculateOi(Tij)
            Dj = self.calculateDj(Tij)
            expBetaCij = np.exp(-Beta*Cij) #pre-calculate an exp(-Beta*cij) matrix for speed
            for i in range(0,N):
                denom = np.sum(Dj*expBetaCij[i,:]) #sigmaj Dj exp(-Beta*Cij)
                Tij2=Oi[i]*(Dj*expBetaCij[i]/denom)
                TPred[i,:]=Tij2 #put answer slice back in return array
            #end for i
        #end for r
        finishtime = time.time()
        #print("SingleDest: benchmarkRun ",finishtime-starttime," seconds")
        return (TPred,finishtime-starttime)

###############################################################################
