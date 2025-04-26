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
    HYPER_PARAM    = 7

class MODEL_TYPE(Enum):
    DNN         = 1
    CNN         = 2
    RNN         = 3
    PhaseNet    = 4
    CRED        = 5
    MobileNet1D = 6


## This class has all the configurations that control the scripts
class Config:
    def __init__(self):
        
        #set program mode
        self.MODE               = MODE_TYPE.ALL
        self.MODEL_TYPE         = MODEL_TYPE.CNN
        # File paths
        self.ORIGINAL_DB_FILE   = "/Users/user/Library/CloudStorage/OneDrive-MasseyUniversity/Technical-Work/databackup/waveforms.hdf5"
        #self.ORIGINAL_DB_FILE  = "data/waveforms_new.hdf5"
        self.METADATA_PATH      = "data/metadata.csv"
        self.MODEL_FILE_NAME    = "models/cnn_20250401_0216_9867.pt_ts.pt" # default model name : model_default.pt. If this is changed, new name will considered as the model_name for testing
        self.MODEL_PATH         = "models/"

        # Below parameters are used in extract_db script to extract certain window in database
        self.DATABASE_FILE  = "data/waveform_2s_data.hdf5" # Overide if file alreay exist
        self.ORIGINAL_SAMPLING_RATE = 50 # Most of the data points are in this category. Hence choosing as the base sampling rate
        self.TRAINING_WINDOW        = 2 # in seconds
        self.BASE_SAMPLING_RATE     = 50
        self.SHIFT_WINDOW           = 10
        self.DATA_EXTRACTED_FILE    = f"data/waveform_2s_data.hdf5"
        

        self.TEST_DATA              = "data/test_data"
        self.TRAIN_DATA             = "data/train_data"

        # Improve the verbosity
        self.ENABLE_PRINT           = 0

        # Calculated parameters
        self.SAMPLE_WINDOW_SIZE = self.BASE_SAMPLING_RATE * self.TRAINING_WINDOW

        # EdgeImpulse support
        self.EDGE_IMPULSE_CSV_PATH = "data/EdgeImpulseCSV/"

        self.TEST_DATA_SPLIT_RATIO = 0.8
        self.IS_SPLIT_DATA         = True

        # ML model settings
        self.BATCH_SIZE = 64

        self.CSV_FILE   = "data/model_details.csv"
        
        # EQTest configs
        self.EQTEST_MODEL_CSV = "data/eqtest_models.csv"
    

    def argParser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--shift_window', type=float, help='Window to be changed shifted from p wave pick')
        args = parser.parse_args()

        self.SHIFT_WINDOW         = args.shift_window if args.shift_window is not None else self.SHIFT_WINDOW

        # UNET parameters
        self.UNET_INPUT_SIZE = 3
        self.UNET_OUTPUT_SIZE = 3


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

        # CNN Model size parameters
        self.conv1_size            = 16
        self.conv2_size            = 16
        self.conv3_size            = 16
        self.fc1_size              = 16
        self.fc2_size              = 16
        self.kernal_size1           = 4
        self.kernal_size2           = 4
        self.kernal_size3           = 4

        self.dropout1              = 0.1
        self.dropout2              = 0.1
        self.dropout3              = 0.1

        self.l2_decay              = 1e-3 
        self.val_acc               = None


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

        parser.add_argument('--conv1_size', type=float, help='size of the conv1 layer')
        parser.add_argument('--conv2_size', type=float, help='size of the conv2 layer')
        parser.add_argument('--conv3_size', type=float, help='size of the conv3 layer')
        parser.add_argument('--fc1_size', type=float, help='size of the fully connected layer 1')
        parser.add_argument('--fc2_size', type=float, help='size of the fully connected layer 2')

        parser.add_argument('--kernal_size1', type=float, help='size of the kernal size of the conv1 layers')
        parser.add_argument('--kernal_size2', type=float, help='size of the kernal size of the conv2 layers')
        parser.add_argument('--kernal_size3', type=float, help='size of the kernal size of the conv3 layers')
        
        parser.add_argument('--dropout1', type=float, help='dropout layer 1')
        parser.add_argument('--dropout2', type=float, help='dropout layer 2')
        parser.add_argument('--dropout3', type=float, help='dropout layer 3')

        parser.add_argument('--l2_decay', type=float, help='L2 weight decay')


        args = parser.parse_args()

        self.learning_rate   = args.learning_rate   if args.learning_rate is not None else self.learning_rate
        self.epoch_count     = args.epoch_count     if args.epoch_count is not None else self.epoch_count
        self.batch_size      = args.batch_size      if args.batch_size is not None else self.batch_size

        self.adam_beta1     = args.adam_beta1 if args.adam_beta1 is not None else self.adam_beta1
        self.adam_beta2     = args.adam_beta2 if args.adam_beta2 is not None else self.adam_beta2
        self.adam_gamma     = args.adam_gamma if args.adam_gamma is not None else self.adam_gamma

        self.detection_threshold = args.detection_threshold if args.detection_threshold is not None else self.detection_threshold

        self.conv1_size     = int(args.conv1_size) if args.conv1_size is not None else self.conv1_size
        self.conv2_size     = int(args.conv2_size) if args.conv2_size is not None else self.conv2_size
        self.conv3_size     = int(args.conv3_size) if args.conv3_size is not None else self.conv3_size
        self.fc1_size       = int(args.fc1_size) if args.fc1_size is not None else self.fc1_size
        self.fc2_size       = int(args.fc2_size) if args.fc2_size is not None else self.fc2_size

        self.kernal_size1    = int(args.kernal_size1) if args.kernal_size1 is not None else self.kernal_size1
        self.kernal_size2    = int(args.kernal_size2) if args.kernal_size2 is not None else self.kernal_size2
        self.kernal_size3    = int(args.kernal_size3) if args.kernal_size3 is not None else self.kernal_size3
        
        self.dropout1       = float(args.dropout1) if args.dropout1 is not None else self.dropout1
        self.dropout2       = float(args.dropout2) if args.dropout2 is not None else self.dropout2
        self.dropout3       = float(args.dropout3) if args.dropout3 is not None else self.dropout3

        self.l2_decay       = float(args.l2_decay) if args.l2_decay is not None else self.l2_decay

        print(f"Training Hyperparameter : Learning Rate = {self.learning_rate}, Epoch count = {self.epoch_count}, Batch Size = {self.batch_size}, conv1 = {self.conv1_size}, conv2 = {self.conv2_size}, conv3 = {self.conv3_size}, fc1 {self.fc1_size}, fc2 {self.fc2_size} , CNN filter1 = {self.kernal_size1}, CNN filter2= {self.kernal_size2}, CNN filter3= {self.kernal_size3}") # Add others upon on the requirement
