from utils import *
from database_op import *
from config import Config, MODE_TYPE, MODEL_TYPE
from dataprep import pre_proc_data



def plot_waveform(waveform, title):
    """
    Plots a 3-component waveform (X, Y, Z).
    
    Parameters:
    waveform (numpy array): Shape (3, 100) representing 3 channels.
    title (str): Title of the plot.
    """
    plt.figure(figsize=(10, 6))
    
    # Plot each component
    plt.plot(waveform[0], label="X-axis", color="r")  # Red for X
    plt.plot(waveform[1], label="Y-axis", color="g")  # Green for Y
    plt.plot(waveform[2], label="Z-axis", color="b")  # Blue for Z

    plt.title(title)
    plt.xlabel("Time (samples)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.show()


# def single_pred(model, test_vtr):   
#    model.eval()
#    single_sample = test_vtr[0]
#    single_sample_tensor = torch.tensor(single_sample, dtype=torch.float32)
#    single_sample_tensor = single_sample_tensor.unsqueeze(0)
#    with torch.no_grad():
#       prediction = model(single_sample_tensor)
#    predicted_class = torch.argmax(prediction, dim=1).item()
#    print(f"Predicted Class: {predicted_class}")


def general_eval(): 
   cfg  = Config()
   nncfg = NNCFG()
   nncfg.argParser()

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

   model = torch.jit.load(cfg.MODEL_FILE_NAME)
   model.eval()

   # with torch.no_grad():  # Disable gradients during inference
   #    predictions = model(test_tensor)

   TP, TN, FP, FN = 0, 0, 0 , 0

   with torch.no_grad():  # Disable gradients during inference
      for i, sample in enumerate(test_vtr):
         sample_tensor = torch.tensor(sample, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
         prediction = model(sample_tensor)

         predicted_class = (prediction > nncfg.detection_threshold).item()
         true_label = true_vrt[i]

         if predicted_class == 1 and true_label == 1:
               TP += 1  # True Positive
         elif predicted_class == 0 and true_label == 0:
               TN += 1  # True Negative
         elif predicted_class == 1 and true_label == 0:
               FP += 1  # False Positive
         elif predicted_class == 0 and true_label == 1:
               FN += 1  # False Negative

   # true_tensor = torch.tensor(true_vrt, dtype=torch.long) 
   # predicted_classes = (predictions  > nncfg.detection_threshold)
   # predicted_classes  = predicted_classes.squeeze()

   # assert (predicted_classes.shape == true_tensor.shape)
   
   # TP = ((predicted_classes == 1) & (true_tensor == 1)).sum().item()  # True Positives
   # TN = ((predicted_classes == 0) & (true_tensor == 0)).sum().item()  # True Negatives
   # FP = ((predicted_classes == 1) & (true_tensor == 0)).sum().item()  # False Positives
   # FN = ((predicted_classes == 0) & (true_tensor == 1)).sum().item()  # False Negatives

   # Calculate Accuracy, Precision, Recall, and F1 Score
   accuracy = 100*((TP + TN) / (TP + TN + FP + FN))
   precision = 100*(TP / (TP + FP)) if (TP + FP) != 0 else 0
   recall = 100*(TP / (TP + FN)) if (TP + FN) != 0 else 0
   f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
   parameters = count_parameters(model)
   
   print(f'Accuracy: {accuracy:.4f}%')
   print(f'Precision: {precision:.4f}%')
   print(f'Recall: {recall:.4f}%')
   print(f'F1 Score: {f1:.4f}%')
   print(f'Parameters: {parameters}')
   
   ## Total samples in each category
   # num_p = len(p_data)
   # num_s = len(s_data)
   # num_noise = len(noise_data)

   # # Get indices for each category
   # p_indices = np.arange(0, num_p)  # P-waves
   # s_indices = np.arange(num_p, num_p + num_s)  # S-waves
   # noise_indices = np.arange(num_p + num_s, num_p + num_s + num_noise)  # Noise

   # # Find incorrect classifications
   # incorrect_p = (predicted_classes[p_indices] == 0).sum()  # P-waves misclassified as noise
   # incorrect_s = (predicted_classes[s_indices] == 0).sum()  # S-waves misclassified as noise
   # incorrect_noise = (predicted_classes[noise_indices] == 1).sum()  # Noise misclassified as earthquake

   # # Print results
   # print(f"Incorrectly classified P-waves: {incorrect_p}/{num_p}")
   # print(f"Incorrectly classified S-waves: {incorrect_s}/{num_s}")
   # print(f"Incorrectly classified Noise: {incorrect_noise}/{num_noise}")

   # print(f"P wave detection accuracy : {((num_p - incorrect_p)/num_p)*100} %")
   # print(f"S wave detection accuracy : {((num_s - incorrect_s)/num_s)*100} %")
   # print(f"Noise wave detection accuracy : {((num_noise - incorrect_noise)/num_noise)*100} %")


def manual_threshold():
   cfg  = Config()
   nncfg = NNCFG()
   nncfg.argParser()
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

   model = torch.jit.load(cfg.MODEL_FILE_NAME+"_ts.pt")
   model.eval()

   with torch.no_grad():  # Disable gradients during inference
      predictions = model(test_tensor)

   true_tensor = torch.tensor(true_vrt, dtype=torch.long) 
   predicted_classes = torch.argmax(predictions, dim=1)
   assert (predicted_classes.shape == true_tensor.shape)
   
   eval_test_report(model, true_tensor, predicted_classes)


   #Manual threshold
   for x in np.arange(0.1, 1.0, 0.01):
      nncfg.detection_threshold = x
      predicted_classes = (torch.softmax(predictions, dim=1)[:, 1] >= nncfg.detection_threshold).int()      
      assert (predicted_classes.shape == true_tensor.shape)
      res = test_report(cfg, nncfg, model, true_tensor, predicted_classes)
   

   ## Total samples in each category
   num_p = len(p_data)
   num_s = len(s_data)
   num_noise = len(noise_data)

   # Get indices for each category
   p_indices = np.arange(0, num_p)  # P-waves
   s_indices = np.arange(num_p, num_p + num_s)  # S-waves
   noise_indices = np.arange(num_p + num_s, num_p + num_s + num_noise)  # Noise

   # Find incorrect classifications
   incorrect_p = (predicted_classes[p_indices] == 0).sum()  # P-waves misclassified as noise
   incorrect_s = (predicted_classes[s_indices] == 0).sum()  # S-waves misclassified as noise
   incorrect_noise = (predicted_classes[noise_indices] == 1).sum()  # Noise misclassified as earthquake

   # Print results
   print(f"Incorrectly classified P-waves: {incorrect_p}/{num_p}")
   print(f"Incorrectly classified S-waves: {incorrect_s}/{num_s}")
   print(f"Incorrectly classified Noise: {incorrect_noise}/{num_noise}")

general_eval()
