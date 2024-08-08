import os
import os.path
import zipfile
import urllib.request
import ssl
#THIS IS VERY BAD - urllib request throws bad cert on OSF.io otherwise
ssl._create_default_https_context = ssl._create_unverified_context
#import numpy as np
from shutil import copyfile
#from globals import *
#from utils import loadZoneLookup #moved to zonecodes
#from zonecodes import ZoneCodes
#from utils import generateTripsMatrix
#from utils import loadMatrix, saveMatrix, loadQUANTMatrix, loadQUANTCSV

url_QUANT2_TObs1_qcs = 'https://osf.io/ga9m3/download'
url_QUANT2_TObs2_qcs = 'https://osf.io/nfepz/download'
url_QUANT2_TObs3_qcs = 'https://osf.io/at9vc/download'
url_QUANT2_TijRoad_qcs = 'https://osf.io/an2yv/download' #TPred_Q3_1
url_QUANT2_TijBus_qcs = 'https://osf.io/vjd7x/download' #TPred_Q3_2
url_QUANT2_TijRail_qcs = 'https://osf.io/4j68z/download' #TPred_Q3_3
url_QUANT2_CijRoadMin_qcs = 'https://osf.io/u2mz6/download' #dis_roads_min
url_QUANT2_CijBusMin_qcs = 'https://osf.io/bd4s2/download' #dis_bus_min
url_QUANT2_CijGBRailMin_qcs = 'https://osf.io/gq8z7/download' #dis_gbrail_min
#url_QUANT2_ZoneCodesText = 'https://osf.io/hu7bd/download'
url_QUANT2_ZoneCodesXML = 'https://osf.io/sjkm5/download'
url_QUANT2_ZoneCodesXSD = 'https://osf.io/yfbrd/download'
url_QUANT2_CrowflyVertexRoadKM_qcs = 'https://osf.io/bs6wq/download'
url_QUANT2_CrowflyVertexBusKM_qcs = 'https://osf.io/ktpze/download'
url_QUANT2_CrowflyVertexGBRailKM_qcs = 'https://osf.io/56ujs/download'

modelRunsDir = '../inputs/model-runs'

ZoneCodesFilenameCSV = 'ZoneCodesText.csv'
ZoneCodesFilenameXML = 'EWS_ZoneCodes.xml'
ZoneCodesFilenameXSD = 'EWS_ZoneCodes.xsd'
#matrix filenames - these are C# matrix dumps direct from QUANT
TObs31Filename = 'TObs_1.bin' #3 mode
TObs32Filename = 'TObs_2.bin'
TObs33Filename = 'TObs_3.bin'
TPred31Filename = 'TPred_Q3_1.bin'
TPred32Filename = 'TPred_Q3_2.bin'
TPred33Filename = 'TPred_Q3_3.bin'
#cost matrix names
disRoadMinFilename = 'dis_roads_min.bin' #these are C# matrix dumps
disBusMinFilename = 'dis_bus_min.bin'
disGBRailMinFilename = 'dis_gbrail_min.bin'
#distance KM files
disCrowflyVertexRoadKMFilename = 'dis_crowfly_vertex_roads_KM.bin'
disCrowflyVertexBusKMFilename = 'dis_crowfly_vertex_bus_KM.bin'
disCrowflyVertexRailKMFilename = 'dis_crowfly_vertex_gbrail_KM.bin'


"""
Utility to check for the existence of a plain file and download it from the given url if
it does not exist.
@param localFilename the name of the file in the localDir dir
@param localDir the dir containing localFilename
@param the url to download it from if not present on the current file system
"""
def ensurePlainFile(localFilename, localDir, url):
    if os.path.isfile(os.path.join(localDir, localFilename)):
        print('install.py:',localFilename,' exists - skipping')
    else:
        print('install.py: ',localFilename,' downloading from ',url)
        path = os.path.join(localDir,localFilename)
        urllib.request.urlretrieve(url, path)
        print('install.py: created file ',localFilename,' in',localDir)
###


#MAIN PROGRAM

#make sure we have a model runs dir
if not os.path.exists(modelRunsDir):
    os.mkdir(modelRunsDir)

#TObs1/2/3
ensurePlainFile(TObs31Filename,modelRunsDir,url_QUANT2_TObs1_qcs)
ensurePlainFile(TObs32Filename,modelRunsDir,url_QUANT2_TObs2_qcs)
ensurePlainFile(TObs33Filename,modelRunsDir,url_QUANT2_TObs3_qcs)
#TPred_Q3_1/2/3
ensurePlainFile(TPred31Filename,modelRunsDir,url_QUANT2_TijRoad_qcs)
ensurePlainFile(TPred32Filename,modelRunsDir,url_QUANT2_TijBus_qcs)
ensurePlainFile(TPred33Filename,modelRunsDir,url_QUANT2_TijRail_qcs)
#dis_road/bus/gbrail_min
ensurePlainFile(disRoadMinFilename,modelRunsDir,url_QUANT2_CijRoadMin_qcs)
ensurePlainFile(disBusMinFilename,modelRunsDir,url_QUANT2_CijBusMin_qcs)
ensurePlainFile(disGBRailMinFilename,modelRunsDir,url_QUANT2_CijGBRailMin_qcs)
#dis crowfly vertex road/bus/rail KM
ensurePlainFile(disCrowflyVertexRoadKMFilename,modelRunsDir,url_QUANT2_CrowflyVertexRoadKM_qcs)
ensurePlainFile(disCrowflyVertexBusKMFilename,modelRunsDir,url_QUANT2_CrowflyVertexBusKM_qcs)
ensurePlainFile(disCrowflyVertexRailKMFilename,modelRunsDir,url_QUANT2_CrowflyVertexGBRailKM_qcs)
#and finally the zonecodes files
ensurePlainFile(ZoneCodesFilenameXML,modelRunsDir,url_QUANT2_ZoneCodesXML)
ensurePlainFile(ZoneCodesFilenameXSD,modelRunsDir,url_QUANT2_ZoneCodesXSD)

