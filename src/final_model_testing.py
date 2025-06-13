from utils import *
from database_op import *
from config import Config, MODE_TYPE, MODEL_TYPE
from dataprep import pre_proc_data
from plots import plot_accuracy_vs_metadata, plot_roc_curve

from ptflops import get_model_complexity_info


def filter_by_year(metadata, data, max_year=2019):
      filtered_data = []
      filtered_metadata = []
      for i, meta in enumerate(metadata):
         event_id = str(meta[0])
         try:
            year = int(event_id[:4])
         except Exception:
            continue
         if year >= max_year:
            filtered_data.append(data[i])
            filtered_metadata.append(meta)
      return np.array(filtered_data), filtered_metadata


# This script can be used to test and plot different aspects of the final model 
def final_test(): 
   
   cfg = Config()
   nncfg = NNCFG()
   nncfg.argParser()

   model_file_name = "models/cnn_20250613_1327_1922.pt"
   cfg.MODEL_FILE_NAME = model_file_name

   nncfg.model_id =  os.path.splitext(os.path.basename(cfg.MODEL_FILE_NAME))[0]

   if not os.path.isfile(cfg.MODEL_FILE_NAME):
      raise ValueError(f"No model found as :{cfg.MODEL_FILE_NAME}")
   
   model = None
   model = torch.jit.load(cfg.MODEL_FILE_NAME)
   model.eval()

   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

   hdf5_file = h5py.File(cfg.TEST_DATA, 'r')
   p_data, noise_data, p_metadata = getWaveData(cfg, hdf5_file)

   p_data      = np.array(p_data)
   noise_data  = np.array(noise_data)

   p_data_orig = p_data
   p_data      = normalize_data(p_data)
   noise_data  = normalize_data(noise_data)

   # Filter out data after the year 2019 for both p and noise data
   #p_data, p_metadata = filter_by_year(p_metadata, p_data, max_year=2019)

   # Balance: get same number of noise samples as filtered p_data
   n_p = len(p_data)
   if len(noise_data) > n_p:
       random_state = np.random.RandomState(42)  # For reproducibility
       noise_indices = random_state.choice(len(noise_data), n_p, replace=False)
       noise_data = noise_data[noise_indices]


   test_val_split_ratio = 0.5
   random_state = np.random.RandomState(42)  # Set a fixed random seed for consistency

   all_indices_p = np.arange(len(p_data))
   random_indices_p = random_state.choice(len(p_data), int(test_val_split_ratio * p_data.shape[0]), replace=False)
   test_indices_p = np.setdiff1d(all_indices_p, random_indices_p)  # Indices not used in validation
   p_data_test = p_data[test_indices_p]
   p_metadata_test = [p_metadata[i] for i in test_indices_p]
   # Get indices for noise data
   all_indices_noise = np.arange(len(noise_data))
   random_indices_noise = random_state.choice(len(noise_data), int(test_val_split_ratio * noise_data.shape[0]), replace=False)
   test_indices_noise = np.setdiff1d(all_indices_noise, random_indices_noise)  # Indices not used in validation
   noise_data_test = noise_data[test_indices_noise]


   #true_vrt    = np.array([1] * len(p_data_test) +[0] * len(noise_data_test))
   #test_vtr    = np.concatenate((p_data_test,  noise_data_test))

   true_vrt    = np.array([1] * len(p_data) +[0] * len(noise_data))
   test_vtr    = np.concatenate((p_data,  noise_data))

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


   TP = ((predicted_classes == 1) & (true_tensor == 1)).sum().item()  # True Positives
   TN = ((predicted_classes == 0) & (true_tensor == 0)).sum().item()  # True Negatives
   FP = ((predicted_classes == 1) & (true_tensor == 0)).sum().item()  # False Positives
   FN = ((predicted_classes == 0) & (true_tensor == 1)).sum().item()  # False Negatives

   # Calculate Accuracy, Precision, Recall, and F1 Score
   accuracy = 100 * ((TP + TN) / (TP + TN + FP + FN))
   precision = 100 * (TP / (TP + FP)) if (TP + FP) != 0 else 0
   recall = 100 * (TP / (TP + FN)) if (TP + FN) != 0 else 0
   f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
   parameters = count_parameters(model)
   #macs, params = get_model_complexity_info(model, (3, cfg.SAMPLE_WINDOW_SIZE), as_strings=True)

   # Print the results
   print(f'Accuracy: {accuracy:.4f}%')
   print(f'Precision: {precision:.4f}%')
   print(f'Recall: {recall:.4f}%')
   print(f'F1 Score: {f1:.4f}%')
   print(f'Parameters: {parameters}')
   
   #acc_vs_metadata_img = cfg.MODEL_PATH + nncfg.model_id + "_acc_metadata.jpg"
   #plot_accuracy_vs_metadata(true_tensor, predicted_classes, p_metadata, acc_vs_metadata_img, p_data_orig)

   plot_roc_curve(true_tensor, predictions, "plots/" + nncfg.model_id + "_roc_curve.jpg")

   #print(f"Plot saved to: {acc_vs_metadata_img}")
   print("Final test completed successfully.")
   

final_test()
