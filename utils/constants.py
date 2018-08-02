from dba import dba

from distances.dtw.dtw import dynamic_time_warping as dtw

from augment import get_weights_average_selected

UNIVARIATE_DATASET_NAMES = ['50words','Adiac','ArrowHead','Beef','BeetleFly',
                            'BirdChicken','Car','CBF','ChlorineConcentration',
                            'CinC_ECG_torso','Coffee','Computers','Cricket_X',
                            'Cricket_Y','Cricket_Z','DiatomSizeReduction',
                            'DistalPhalanxOutlineAgeGroup',
                            'DistalPhalanxOutlineCorrect','DistalPhalanxTW',
                            'Earthquakes','ECG200','ECG5000','ECGFiveDays',
                            'ElectricDevices','FaceAll','FaceFour','FacesUCR',
                            'FISH','FordA','FordB','Gun_Point','Ham',
                            'HandOutlines','Haptics','Herring','InlineSkate',
                            'InsectWingbeatSound','ItalyPowerDemand',
                            'LargeKitchenAppliances','Lighting2','Lighting7',
                            'MALLAT','Meat','MedicalImages',
                            'MiddlePhalanxOutlineAgeGroup',
                            'MiddlePhalanxOutlineCorrect','MiddlePhalanxTW',
                            'MoteStrain','NonInvasiveFatalECG_Thorax1',
                            'NonInvasiveFatalECG_Thorax2','OliveOil','OSULeaf',
                            'PhalangesOutlinesCorrect','Phoneme','Plane',
                            'ProximalPhalanxOutlineAgeGroup',
                            'ProximalPhalanxOutlineCorrect',
                            'ProximalPhalanxTW','RefrigerationDevices',
                            'ScreenType','ShapeletSim','ShapesAll',
                            'SmallKitchenAppliances','SonyAIBORobotSurface',
                            'SonyAIBORobotSurfaceII','StarLightCurves',
                            'Strawberry','SwedishLeaf','Symbols',
                            'synthetic_control','ToeSegmentation1',
                            'ToeSegmentation2','Trace','TwoLeadECG',
                            'Two_Patterns','UWaveGestureLibraryAll',
                            'uWaveGestureLibrary_X','uWaveGestureLibrary_Y',
                            'uWaveGestureLibrary_Z','wafer','Wine',
                            'WordsSynonyms','Worms','WormsTwoClass','yoga']

# UNIVARIATE_DATASET_NAMES = ['BirdChicken','DiatomSizeReduction']

UNIVARIATE_ARCHIVE_NAMES = ['UCR_TS_Archive_2015']

AVERAGING_ALGORITHMS = {'dba':dba}

DISTANCE_ALGORITHMS = {'dtw': dtw}

DTW_PARAMS = {'w':-1} # warping window should be given in percentage (negative means no warping window)

DISTANCE_ALGORITHMS_PARAMS = {'dtw':DTW_PARAMS}

MAX_PROTOTYPES_PER_CLASS = 5

WEIGHTS_METHODS = {'as':get_weights_average_selected }
