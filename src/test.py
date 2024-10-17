
from utils import *
from database_op import *
from config import Config, MODE_TYPE, MODEL_TYPE


def test(cfg): 
   print("Runnig for test set")

   model = None

   if cfg.MODEL_TYPE == MODEL_TYPE.DNN:
      model = DNN()
   elif cfg.MODEL_TYPE == MODEL_TYPE.CNN:
      model = PWaveCNN(cfg.SAMPLE_WINDOW_SIZE)

   model.load_state_dict(torch.load(cfg.MODEL_FILE_NAME))
   model.eval()

   hdf5_file = h5py.File("data/test_data", 'r')
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

   correct_predictions = (predicted_classes == true_tensor).sum().item() 
   total_predictions = true_tensor.size(0)
   accuracy = correct_predictions / total_predictions
   print(f"Prediction accuracy: {accuracy * 100:.2f}%")


