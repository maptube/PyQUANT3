"""
debug.py
various utility debugging routines
"""

from scenarios.OneLink import OneLinkLimitR
from models.SingleOrigin import SingleOrigin

"""
Count how many scenarios there are in total for every zone (i)
Output to countscenarios.txt
"""
def debug_countScenarios(radiusKM,N,mode,Lij_rail):
    with open('countscenarios.csv','w') as fd:
        fd.write('zonei,count_'+str(radiusKM)+'KM\n')
        scenarioGenerator = OneLinkLimitR(radiusKM,N,mode,Lij_rail)
        scenarioGenerator.i=0
        scenarioGenerator.j=-1
        count=0
        currenti=0
        while scenarioGenerator.next()!=[]:
            if (scenarioGenerator.i!=currenti):
                fd.write(str(currenti)+','+str(count)+'\n')
                count=0
                currenti=scenarioGenerator.i
            count+=1
        fd.write(str(currenti)+','+str(count)+'\n')
###

"""
Write out Oi and Dj tables to be used with ML programs.
NOTE: this uses TPredicted matrices from the baseline calibration (qm3_baseline).
You could pass in the TObs, but the model is using TPred, so this
makes sense.
NOTE: this obviously depends on you knowing the beta values.
"""        
def debug_OiDjTable(TPred):

    Oi_road = SingleOrigin.calculateOi(TPred[0])
    Oi_bus = SingleOrigin.calculateOi(TPred[1])
    Oi_rail = SingleOrigin.calculateOi(TPred[2])
    Dj_road = SingleOrigin.calculateDj(TPred[0])
    Dj_bus = SingleOrigin.calculateDj(TPred[1])
    Dj_rail = SingleOrigin.calculateDj(TPred[2])

    with open('OiDjTable.csv','w') as fd:
        fd.write('zonei,Oi_road,Oi_bus,Oi_rail,Oi_all,Dj_road,Dj_bus,Dj_rail,Dj_all\n')
        N = len(Oi_road)
        for i in range(0,N):
            fd.write(
                str(i)
                +','+str(Oi_road[i])+','+str(Oi_bus[i])+','+str(Oi_rail[i])+','+str(Oi_road[i]+Oi_bus[i]+Oi_rail[i])
                +','+str(Dj_road[i])+','+str(Dj_bus[i])+','+str(Dj_rail[i])+','+str(Dj_road[i]+Dj_bus[i]+Dj_rail[i])
                +'\n'
            )
###
