"""
unit test for testing the impacts code
python -m unittest discover
"""

import unittest
import os
from pathlib import Path
import yaml
import time

from utils import loadQUANTMatrix, loadQUANTMatrixFAST
from impacts.ImpactStatistics import ImpactStatistics
from models.DirectNetworkChange import DirectNetworkChange

#todo:
#NOTE: the method MUST be called test_xxx, otherwise it won't be run!
class Test_ImpactStatisticsMethods(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        configuration = yaml.safe_load(open("appsettings.yaml")) #note change of dir
        input_folder=Path(configuration["dirs"]["LocalInputsDir"])
        output_folder=Path(configuration["dirs"]["LocalOutputsDir"])
        ModelRunsDirName = configuration["dirs"]["LocalModelRunsDir"]
        ModelRunsDir = input_folder.joinpath(ModelRunsDirName) #NOTE: this corrupts input folder????
        TijObsRoadFilename = configuration["matrices"]["TObs1"]
        TijObsBusFilename = configuration["matrices"]["TObs2"]
        TijObsRailFilename = configuration["matrices"]["TObs3"]
        DisRoadFilename = configuration["matrices"]["dis_roads"]
        DisBusFilename = configuration["matrices"]["dis_buses"]
        DisGBRailFilename = configuration["matrices"]["dis_rail"]
        DisCrowflyKMFilename = configuration["matrices"]["dis_crowfly"]
        DisCrowflyVertexRoadsKMFilename = configuration["matrices"]["dis_crowfly_vertex_roads_KM"]
        DisCrowflyVertexBusKMFilename = configuration["matrices"]["dis_crowfly_vertex_bus_KM"]
        DisCrowflyVertexGBRailKMFilename = configuration["matrices"]["dis_crowfly_vertex_gbrail_KM"]

        #now the actual loading
        self.Tij_Obs_road = loadQUANTMatrixFAST(os.path.join(ModelRunsDir,TijObsRoadFilename))
        self.Tij_Obs_bus = loadQUANTMatrixFAST(os.path.join(ModelRunsDir,TijObsBusFilename))
        self.Tij_Obs_rail = loadQUANTMatrixFAST(os.path.join(ModelRunsDir,TijObsRailFilename))
        #and costs
        self.Cij_road = loadQUANTMatrixFAST(os.path.join(ModelRunsDir,DisRoadFilename))
        self.Cij_bus = loadQUANTMatrixFAST(os.path.join(ModelRunsDir,DisBusFilename))
        self.Cij_rail = loadQUANTMatrixFAST(os.path.join(ModelRunsDir,DisGBRailFilename))
        #and transport KM distances
        self.Lij_road = loadQUANTMatrixFAST(os.path.join(ModelRunsDir,DisCrowflyVertexRoadsKMFilename))
        self.Lij_bus = loadQUANTMatrixFAST(os.path.join(ModelRunsDir,DisCrowflyVertexBusKMFilename))
        self.Lij_rail = loadQUANTMatrixFAST(os.path.join(ModelRunsDir,DisCrowflyVertexGBRailKMFilename))
    ###

    def setUp(self):
        print("setUp")
        self.impacts = ImpactStatistics()
    ###


    def test_one(self):
        print("hello test")
        self.assertTrue(True)
    ###

    def test_computeL_k(self):
        print("test computeL_k")
        
        start_time=time.process_time()
        Lk = self.impacts.computeL_k(
            [self.Tij_Obs_road, self.Tij_Obs_bus, self.Tij_Obs_rail],
            [self.Lij_road, self.Lij_bus, self.Lij_rail])
        #print(Lk)
        end_time = time.process_time()
        print('pyquant3::test_computeL_k '+str(end_time-start_time)+' secs')
        #numpy
        #self.assertListEqual(Lk,[249384316.8240262, 20818796.246591613, 54944873.84144509],"Lk check failure NUMPY")
        #JAX
        self.assertListEqual(Lk,[2.4938435e+08, 20818800.0, 54944868.0],"Lk check failure JAX")
    ###
        
    def test_computeScenarioLinkStatistics(self):
        print("test computeScenarioLinkStatistics")
        #network changes and expected outputs here
        tests = [
            {
                'nc': [ DirectNetworkChange(0,0,1,60) ], #mode(0=road,1=bus,2=rail),i,j,secs
                'exp_depth':[1,0,0], 'exp_KM': [17.99045753479004, 0.0, 0.0], 'exp_savedSecs': [684.5629692077637, 0.0, 0.0]
            },
            {
                'nc': [ DirectNetworkChange(1,0,1,60) ],
                'exp_depth':[0,1,0], 'exp_KM': [0.0, 17.20673370361328, 0.0], 'exp_savedSecs': [0.0, 4051.1920166015625, 0.0]
            },
            {
                'nc': [ DirectNetworkChange(2,0,1,60) ],
                'exp_depth':[0,0,1], 'exp_KM': [0.0, 0.0, 17.282915115356445], 'exp_savedSecs': [0.0, 0.0, 1156.4366912841797]
            },
            {
                'nc': [ DirectNetworkChange(0,0,1,60), DirectNetworkChange(0,0,4,180), DirectNetworkChange(0,0,100,500) ],
                'exp_depth':[3,0,0], 'exp_KM': [46.408573150634766, 0.0, 0.0], 'exp_savedSecs': [1195.6647491455078, 0.0, 0.0]
            },
            {
                'nc': [ DirectNetworkChange(0,0,1,60), DirectNetworkChange(1,0,4,180), DirectNetworkChange(2,0,100,500) ],
                'exp_depth':[1,1,1], 'exp_KM': [17.99045753479004, 16.851823806762695, 11.65621280670166], 'exp_savedSecs': [684.5629692077637, 3358.9686584472656, 708.4066009521484]
            }
        ]
        for test in tests:
            linkDepth_k, linkKM_k, linkSavedSecs_k = self.impacts.computeScenarioLinkStatistics(
                test['nc'], [self.Cij_road,self.Cij_bus,self.Cij_rail], [self.Lij_road,self.Lij_bus,self.Lij_rail])
            print('pyquant3::test_computeScenarioLinkStatistics',linkDepth_k,linkKM_k,linkSavedSecs_k)
            self.assertListEqual(linkDepth_k, test['exp_depth'])
            self.assertListEqual(linkKM_k, test['exp_KM'])
            self.assertListEqual(linkSavedSecs_k, test['exp_savedSecs'])
        #end for
    ###
            
    def test_computeLBar(self):
        print("test computeLBar")
        tests = [
            {
                'nc': [ DirectNetworkChange(0,0,1,60) ], #mode(0=road,1=bus,2=rail),i,j,secs
                'exp_LBar':[17.99045753479004, 0, 0]
            },
            {
                'nc': [ DirectNetworkChange(1,0,1,60) ],
                'exp_LBar':[0, 17.20673370361328, 0]
            },
            {
                'nc': [ DirectNetworkChange(2,0,1,60) ],
                'exp_LBar':[0, 0, 17.282915115356445]
            },
            {
                'nc': [
                    DirectNetworkChange(2,123,456,600), DirectNetworkChange(1,100,101,60), DirectNetworkChange(0,200,204,120), DirectNetworkChange(2,314,612,240),
                    DirectNetworkChange(1,2,3,60), DirectNetworkChange(0,1000,2000,500), DirectNetworkChange(2,8400,8401,600), DirectNetworkChange(1,7190,7199,30)
                ],
                'exp_LBar':[175.66357773542404, 123.72714523474376, 318.7830987294515]
            }
        ]
        for test in tests:
            LBar = self.impacts.computeLBar(test['nc'], [self.Lij_road,self.Lij_bus,self.Lij_rail])
            print('pyquant3::test_computeLBar')
            self.assertListEqual(LBar,test['exp_LBar'])
    ###
            
    #def computeNetworkStatistics(self,Cij1,Cij2):
    #returns nMinus_k, savedSecs_k
            
    #def compute(self,qm3_base: SingleOrigin,qm3: SingleOrigin, dijKM:np.matrix, networkChanges: list):
    #and remember it sets lots of tricky properties on the object


if __name__ == '__main__':
    unittest.main()