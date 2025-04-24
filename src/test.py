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


def test(cfg): 
   
   print("Runnig for test set")
   nncfg = NNCFG()
   nncfg.argParser()

   if cfg.MODEL_FILE_NAME == "models/model_default.pt":
      model_name = getLatestModelName(cfg)
      cfg.MODEL_FILE_NAME = cfg.MODEL_PATH + model_name
   else:
      print(f"Using the model:  {cfg.MODEL_FILE_NAME} for testing")

   if not os.path.isfile(cfg.MODEL_FILE_NAME):
      raise ValueError(f"No model found as :{cfg.MODEL_FILE_NAME}")
   
   model = None

   checkpoint = torch.load(cfg.MODEL_FILE_NAME)
   loadModelConfig(nncfg, checkpoint)

   if cfg.MODEL_TYPE == MODEL_TYPE.DNN:
      model = DNN(model_id=nncfg.model_id)
   elif cfg.MODEL_TYPE == MODEL_TYPE.CNN:
      model = PWaveCNN(
         model_id=nncfg.model_id, 
         window_size=cfg.SAMPLE_WINDOW_SIZE, 
         conv1_filters=nncfg.conv1_size, 
         conv2_filters=nncfg.conv2_size, 
         conv3_filters=nncfg.conv3_size, 
         fc1_neurons=nncfg.fc1_size, 
         fc2_neurons=nncfg.fc2_size, 
         kernel_size1=nncfg.kernal_size1, 
         kernel_size2=nncfg.kernal_size2, 
         kernel_size3=nncfg.kernal_size3, 
         dropout1=nncfg.dropout1, 
         dropout2=nncfg.dropout2, 
         dropout3=nncfg.dropout3
      )
   elif cfg.MODEL_TYPE == MODEL_TYPE.CRED:
      assert("This routine is yet to be implemented")
      model = sbm.PhaseNet(phases="PSN", norm="peak")
   elif cfg.MODEL_TYPE == MODEL_TYPE.UNET:
      model = UNet(model_id=nncfg.model_id, in_channels=cfg.UNET_INPUT_SIZE, out_channels=cfg.UNET_OUTPUT_SIZE)
   else:
      raise ValueError(f"Invalid model type: {cfg.MODEL_TYPE}")
   
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

   test_subset_len = int(0.2 * p_data.shape[0])

   p_data_test = p_data[test_subset_len:]
   s_data_test = s_data[test_subset_len:]
   noise_data_test = noise_data[test_subset_len*2:]

   true_vrt    = np.array([1] * len(p_data_test) + [1] * len(s_data_test) +[0] * len(noise_data_test))
   
   test_vtr    = np.concatenate((p_data_test, s_data_test, noise_data_test))

   # Convert to tensor
   test_tensor = torch.tensor(test_vtr, dtype=torch.float32)

   with torch.no_grad():  # Disable gradients during inference
      predictions = model(test_tensor)

   predicted_classes = (predictions > nncfg.detection_threshold)
   #predicted_classes = torch.argmax(predictions, dim=1)

   #Calculate the accuracy. This is tempory calculation
   true_tensor = torch.tensor(true_vrt, dtype=torch.long) 
   predicted_classes = predicted_classes.squeeze()
   
   assert (predicted_classes.shape == true_tensor.shape)

   res = test_report(cfg, nncfg, model, true_tensor, predicted_classes)
   
   if res == 0:
      print("Testing completed successfully")



