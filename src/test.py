
from utils import *
from database_op import *
from config import Config, MODE_TYPE, MODEL_TYPE

def loadModelConfig(nncfg, checkpoint):

   nncfg.model_id       = checkpoint['model_id']
   nncfg.learning_rate  = checkpoint['learning_rate']
   nncfg.epoch_count    = checkpoint['epoch_count']
   nncfg.batch_size     = checkpoint['batch_size']

   nncfg.training_loss  = checkpoint['training_loss']
   nncfg.optimizer      = checkpoint['optimizer']


def test(cfg): 
   
   print("Runnig for test set")
   nncfg = NNCFG()

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
      model = PWaveCNN(model_id=nncfg.model_id, window_size=cfg.SAMPLE_WINDOW_SIZE)
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



