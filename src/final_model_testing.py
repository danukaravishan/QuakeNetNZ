from utils import *
from database_op import *
from config import Config, MODE_TYPE, MODEL_TYPE
from dataprep import pre_proc_data

def loadModelConfig(nncfg, checkpoint):

   nncfg.model_id       = checkpoint['model_id']
   nncfg.learning_rate  = checkpoint['learning_rate']
   nncfg.epoch_count    = checkpoint['epoch_count']
   nncfg.batch_size     = checkpoint['batch_size']

   nncfg.training_loss  = checkpoint['training_loss']
   nncfg.optimizer      = checkpoint['optimizer']
   nncfg.val_acc  = checkpoint['validation_acc']


def test(): 
   
   print("Runnig for test set")
   cfg  = Config()
   nncfg = NNCFG()
   nncfg.argParser()

   cfg.MODEL_FILE_NAME = "models/cnn_final.pt"

   if not os.path.isfile(cfg.MODEL_FILE_NAME):
      raise ValueError(f"No model found as :{cfg.MODEL_FILE_NAME}")
   
   model = None

   checkpoint = torch.load(cfg.MODEL_FILE_NAME)
   loadModelConfig(nncfg, checkpoint)


   model = PWaveCNN(model_id=nncfg.model_id, window_size=cfg.SAMPLE_WINDOW_SIZE, conv1_filters=nncfg.conv1_size, conv2_filters=nncfg.conv2_size, fc1_neurons=nncfg.fc1_size, kernel_size1=nncfg.kernal_size1, kernel_size2=nncfg.kernal_size2)
   
   model.load_state_dict(checkpoint['model_state_dict'])
   model.eval()

   hdf5_file = h5py.File(cfg.TEST_DATA, 'r')
   p_data, s_data, noise_data = getWaveData(cfg, hdf5_file)

   p_data      = np.array(p_data)
   s_data      = np.array(s_data)
   noise_data  = np.array(noise_data)

   p_data = pre_proc_data(p_data)
   s_data = pre_proc_data(s_data)
   noise_data = pre_proc_data(noise_data)

   true_vrt    = np.array([1] * len(p_data) + [1] * len(s_data) +[0] * len(noise_data))
   
   test_vtr    = np.concatenate((p_data, s_data, noise_data))

   # Convert to tensor
   test_tensor = torch.tensor(test_vtr, dtype=torch.float32)

   with torch.no_grad():  # Disable gradients during inference
      predictions = model(test_tensor)

   predicted_classes = torch.argmax(predictions, dim=1)
   #predicted_classes = ((predictions >= 0.9).int()).squeeze() # Use detection threshold as 0.5 
   
   #Calculate the accuracy. This is tempory calculation
   true_tensor = torch.tensor(true_vrt, dtype=torch.long) 
   
   assert (predicted_classes.shape == true_tensor.shape)

   res = test_report(cfg, nncfg, model, true_tensor, predicted_classes)
   
   if res == 0:
      print("Testing completed successfully")


test()
