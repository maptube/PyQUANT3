"""
next() - use this as the interface to generate a scenario NOTE: it's always the same scenario
findNearestZone - finds nearest zone centroid to the lat/lon of a real network change
loadFromGraphML - load a scenario graphml file and return a list of directnetworkchanges
"""
import pandas as pd
import xml.dom.minidom
from models.DirectNetworkChange import DirectNetworkChange

class FileScenario:
    def __init__(self,filename:str, mode:int, zonecodes:pd.DataFrame):
        self.filename = filename
        self.mode = mode
        self.zonecodes = zonecodes
    ###

    """
    findNearestZone
    Private function to find the nearest zone to the given lat/lon
    This is sub-optimal for a number of reasons, not least that it's using Euclidean
    distance on lat/lon values. The full table scan doesn't exactly make it good
    either. However, it's used once and only for the individual links in a scenario,
    so shouldn't be a performance issue to scan an 8436 entry table about 10 times.
    @returns the zonei of the zone centroid closest to lat,lon
    """
    def findNearestZone(self, lon:float, lat:float) -> int:
        bestNode=-1
        bestdist2=1e10
        for idx, row in self.zonecodes.iterrows():
            lat2 = row['lat']
            lon2 = row['lon']
            dx=lon2-lon
            dy=lat2-lat
            dist2 = (dx*dx)+(dy*dy)
            if dist2<bestdist2:
                bestdist2=dist2
                bestNode=row['zonei']
        ##endif
        return bestNode
    ###
    
    """
    loadFromGraphML
    static load method, returns a list of DirectNetworkChange(s) from a special format graphml file
    @param graphml file, BUT this one need to have additional lat="" and lon="" attributes on the <nodes>
    so that we can link the scenario to the nearest nodes (i.e. QUANT3 scenario)
    @param 0=road, 1-bus, 2=rail
    @returns a list of DirectNetworkChange(s)
    """
    def loadFromGraphML(self) -> list :
        #todo: build lookup of nearest zones
        dnc = []
        xml_doc = xml.dom.minidom.parse(self.filename)
        nearestZone = {} #dict(string,int), link between nodes in the graphml and their nearest zone centroids
        nodes = xml_doc.getElementsByTagName('node')
        for node in nodes:
            #<node id="CHH" code="" lines="E" lon="0.128262928" lat="51.56802985" name="Chadwell Heath Station" />
            id = node.getAttribute('id')
            strLon = node.getAttribute('lon')
            strLat = node.getAttribute('lat')
            lon = float(strLon)
            lat = float(strLat)
            #and add it to a dictionary lookup
            areakey = self.findNearestZone(lon, lat)
            nearestZone[id]=areakey
            print("loadFromGraphML Node: ",id,strLon,strLat," -> ",areakey)
        #end for

        #now the links
        edges = xml_doc.getElementsByTagName('edge')
        for edge in edges:
            source = edge.getAttribute('source')
            target = edge.getAttribute('target')
            nodeData = edge.getElementsByTagName('data')[0]
            strWeight = nodeData.firstChild #assumes it's a text node inside e.g. <data key="weight">250</data>
            seconds = float(strWeight.nodeValue)
            print('Edge:',source,target,seconds,"mode=",self.mode)

            #matching work
            sourceI = nearestZone[source]
            targetI = nearestZone[target]
            newDNC = DirectNetworkChange(self.mode, sourceI, targetI, seconds)
            dnc.append(newDNC)
        #end for

        return dnc
    
    ################################################################################

    """
    next()
    This is the entry point to return the list of changes, as loaded from a graphml file.
    This is slightly strange compared to OneLink and NLink, which are designed to run
    thousands of random scenarios, but this enables a single reference scenario to be
    run via the same interface.
    Future - you could add the ability to run a list of files through it, or alter
    the scenario in some randomised way i.e. first run is no changes, then it starts
    to alter the scenario.
    """
    def next(self) -> list:
        return self.loadFromGraphML()