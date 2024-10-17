from enum import Enum

## Main modes of the program
class MODE_TYPE(Enum):
    IDLE           = 1
    TRAIN          = 2
    PREDICT        = 3
    MODEL_ACCURACY = 4
    EXTRACT_DATA   = 5
    SPLIT_DATA     = 6

class MODEL_TYPE(Enum):
    DNN = 1
    CNN = 2
    RNN = 3

## This class has all the configurations that control the scripts
class Config:
    def __init__(self):
        
        #set program mode
        self.MODE               = MODE_TYPE.PREDICT

        self.MODEL_TYPE         = MODEL_TYPE.CNN
        # File paths
        self.ORIGINAL_DB_FILE   = "/Users/user/Desktop/Temp/waveforms.hdf5"
        #self.ORIGINAL_DB_FILE   = "data/waveforms_new.hdf5"
        self.METADATA_PATH      = "data/metadata.csv"
        self.MODEL_FILE_NAME    = "models/mode_cnn_test.pt"

        # Below parameters are used in extract_db script to extract certain window in database
        self.DATABASE_FILE  = "data/waveforms_4s_new_full.hdf5" # Overide if file alreay exist
        self.ORIGINAL_SAMPLING_RATE = 50 # Most of the data points are in this category. Hence choosing as the base sampling rate
        self.TRAINING_WINDOW        = 4 # in seconds
        self.BASE_SAMPLING_RATE     = 50

        self.TRAINING_WINDOW        = 4 # in seconds

        # Improve the verbosity
        self.ENABLE_PRINT       = 0

        # Calculated parameters
        self.SAMPLE_WINDOW_SIZE = self.BASE_SAMPLING_RATE * self.TRAINING_WINDOW

        # EdgeImpulse support
        self.EDGE_IMPULSE_CSV_PATH = "data/EdgeImpulseCSV/"

        self.TEST_DATA_SPLIT_RATIO = 0.2
        self.IS_SPLIT_DATA         = True