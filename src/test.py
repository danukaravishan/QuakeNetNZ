from utils import *
from database_op import *
from config import Config, MODE_TYPE, MODEL_TYPE
from dataprep import pre_proc_data, normalize_data, apply_wavelet_denoise, clean_low_pga
import torch.nn.functional as F

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
   else:
      print(f"Using the model:  {cfg.MODEL_FILE_NAME} for testing")
   
   nncfg.model_id =  os.path.splitext(os.path.basename(cfg.MODEL_FILE_NAME))[0]

   if not os.path.isfile(cfg.MODEL_FILE_NAME):
      raise ValueError(f"No model found as :{cfg.MODEL_FILE_NAME}")
   
   model = None
   model = torch.jit.load(cfg.MODEL_FILE_NAME)
   model.eval()

   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

   hdf5_file = h5py.File(cfg.TEST_DATA, 'r')
   p_data, noise_data = getWaveData(cfg, hdf5_file)

   p_data      = np.array(p_data)
   noise_data  = np.array(noise_data)

   p_data      = normalize_data(p_data)
   noise_data  = normalize_data(noise_data)

   test_val_split_ratio = 0.5

   random_state = np.random.RandomState(42)  # Set a fixed random seed for consistency

   all_indices_p = np.arange(len(p_data))
   random_indices_p = random_state.choice(len(p_data), int(test_val_split_ratio * p_data.shape[0]), replace=False)
   test_indices_p = np.setdiff1d(all_indices_p, random_indices_p)  # Indices not used in validation
   p_data_test = p_data[test_indices_p]

   # Get indices for noise data
   all_indices_noise = np.arange(len(noise_data))
   random_indices_noise = random_state.choice(len(noise_data), int(test_val_split_ratio * noise_data.shape[0]), replace=False)
   test_indices_noise = np.setdiff1d(all_indices_noise, random_indices_noise)  # Indices not used in validation
   noise_data_test = noise_data[test_indices_noise]

   true_vrt    = np.array([1] * len(p_data_test) +[0] * len(noise_data_test))
   test_vtr    = np.concatenate((p_data_test,  noise_data_test))

   # Convert to tensor
   test_tensor = torch.tensor(test_vtr, dtype=torch.float32)
   test_tensor = test_tensor.to(device)  # Move to GPU if available
   
   with torch.no_grad():  # Disable gradients during inference
      predictions = model(test_tensor)

   true_tensor = torch.tensor(true_vrt, dtype=torch.long).to(device)  # Move to GPU if available

   if cfg.MODEL_TYPE == MODEL_TYPE.TFEQ:
      output_prob = F.softmax(predictions, dim=1)[:,1]
      predicted_classes = predictions.max(1, keepdim=True)[1]
   else:
      predicted_classes = (predictions > nncfg.detection_threshold)
      predicted_classes = predicted_classes.to(device)  # Move to GPU if available
      predicted_classes = predicted_classes.squeeze()
      
      assert (predicted_classes.shape == true_tensor.shape)

   res = test_report(cfg, nncfg, model, true_tensor, predicted_classes)
   
   if res == 0:
      print("Testing completed successfully")