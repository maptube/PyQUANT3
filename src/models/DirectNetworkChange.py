"""
DirectNetworkChange
Used for passing network changes to models. It's just a list
"""

class DirectNetworkChange:
    
    """
    Constructor
    @param "mode" 0=road, 1=bus, 2=rail
    @param "originZonei" zone number of origin
    @param "destinationZonei" zone number of destination
    @param "seconds" time of new link between origina and destination i in seconds
    """
    def __init__(self, mode, originZonei, destinationZonei, seconds):
        self.mode=mode #int 0=road, 1=bus, 2=rail
        self.originZonei = originZonei
        self.destinationZonei = destinationZonei
        self.absoluteTimeSecs = seconds
    
###############################################################################
