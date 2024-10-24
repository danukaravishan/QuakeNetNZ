from enum import Enum
import argparse

## Main modes of the program
class MODE_TYPE(Enum):
    IDLE           = 1
    TRAIN          = 2
    PREDICT        = 3
    ALL            = 4
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
        self.MODE               = MODE_TYPE.ALL

        self.MODEL_TYPE         = MODEL_TYPE.CNN
        # File paths
        self.ORIGINAL_DB_FILE   = "/Users/user/Desktop/Temp/waveforms.hdf5"
        #self.ORIGINAL_DB_FILE  = "data/waveforms_new.hdf5"
        self.METADATA_PATH      = "data/metadata.csv"
        self.MODEL_FILE_NAME    = "models/model_default.pt" # default model name : model_default.pt. If this is changed, new name will considered as the model_name for testing
        self.MODEL_PATH         = "models/"

        # Below parameters are used in extract_db script to extract certain window in database
        self.DATABASE_FILE  = "data/waveforms_4s_new_full.hdf5" # Overide if file alreay exist
        self.ORIGINAL_SAMPLING_RATE = 50 # Most of the data points are in this category. Hence choosing as the base sampling rate
        self.TRAINING_WINDOW        = 4 # in seconds
        self.BASE_SAMPLING_RATE     = 50

        self.TEST_DATA              = "data/test_data"
        self.TRAIN_DATA             = "data/train_data"

        self.TRAINING_WINDOW        = 4 # in seconds

        # Improve the verbosity
        self.ENABLE_PRINT           = 0

        # Calculated parameters
        self.SAMPLE_WINDOW_SIZE = self.BASE_SAMPLING_RATE * self.TRAINING_WINDOW

        # EdgeImpulse support
        self.EDGE_IMPULSE_CSV_PATH = "data/EdgeImpulseCSV/"

        self.TEST_DATA_SPLIT_RATIO = 0.2
        self.IS_SPLIT_DATA         = True

        # ML model settings
        self.BATCH_SIZE = 64

        self.CSV_FILE   = "data/model_details.csv"


class NNCFG:
    def __init__(self):
        self.learning_rate          = 0.001
        self.epoch_count            = 2
        self.batch_size             = 32

        self.adam_beta1             = 0.1
        self.adam_beta2             = 0.1
        self.adam_gamma             = 0.1

        self.detection_threshold    = 0.5

        # Dynamic variables
        self.training_loss          = None
        self.optimizer              = None
        self.model_id               = None



    def argParser(self):
        parser = argparse.ArgumentParser()

        # Add arguments
        parser.add_argument('--learning_rate', type=float, help='Learning rate of the NN (int)')
        parser.add_argument('--epoch_count', type=int, help='Number of epoches')
        parser.add_argument('--batch_size', type=int, help='Batch size')

        parser.add_argument('--adam_beta1', type=float, help='Beta 1 of Adam optimizer')
        parser.add_argument('--adam_beta2', type=float, help='Beta 2 of Adam optimizer')
        parser.add_argument('--adam_gamma', type=float, help='Gamma of Adam optimizer')
        parser.add_argument('--detection_threshold', type=float, help='Detection threshold of when one output neuron exist')

        args = parser.parse_args()

        self.learning_rate   = args.learning_rate   if args.learning_rate is not None else self.learning_rate
        self.epoch_count     = args.epoch_count     if args.epoch_count is not None else self.epoch_count
        self.batch_size      = args.batch_size      if args.batch_size is not None else self.batch_size

        self.adam_beta1     = args.adam_beta1 if args.adam_beta1 is not None else self.adam_beta1
        self.adam_beta2     = args.adam_beta2 if args.adam_beta2 is not None else self.adam_beta2
        self.adam_gamma     = args.adam_gamma if args.adam_gamma is not None else self.adam_gamma

        self.detection_threshold = args.detection_threshold if args.detection_threshold is not None else self.detection_threshold

        print(f"Training Hyperparameter : Learning Rate = {self.learning_rate}, Epoch count = {self.epoch_count}, Batch Size = {self.batch_size}") # Add others upon on the requirement