"""
NetworkUtils.py

Utilities to do with networks.

linkKMPerHourToSeconds - given an origin and destination zonei, a distance matrix (KM) and a required speed,
calculate what the runlink time needs to be to traverse i->j at the desired speed

"""


#todo: speed conversion consts here e.g. MPHtoKPH

class NetworkUtils:

    """
    linkKMPerHourToSeconds
    Given an origin and destination zonei (and j), a distance matrix in KM and the required speed
    to traverse i->j in KM per Hour, calculate how long that should take.
    OK, it's just Time=Distance / Velocity
    @param i origin zone index number
    @param j destination zone index number
    @param DisKM (matrix) distances for all i to j zones from teh model (the crowfly vertex KM matrix for the right mode)
    @param KMPerHour the velocity in KM per hour - use the conversion factors if working in MPH or ms-1
    @returns the time to transit the i->j link in seconds for this speed
    """
    @staticmethod
    def linkKMPerHourToSeconds(i,j,DisKM,KMPerHour):
        distKM = DisKM[i,j]
        return  (distKM / KMPerHour) * 3600.0 #time in hours converted to seconds
    
    ################################################################################

    

