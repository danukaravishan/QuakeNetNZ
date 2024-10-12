from utils import *
from database_op import *
from config import Config, MODE_TYPE, MODEL_TYPE


def train(cfg):
      
   hdf5_file = h5py.File("data/train_data", 'r')
   p_data, s_data, noise_data = getWaveData(cfg, hdf5_file)

   print("\nCompleted loading data\n")
   
   # Data preparation
   p_data = np.array(p_data)
   s_data = np.array(s_data)
   noise_data = np.array(noise_data)

   positive_data = np.concatenate((p_data , s_data))

   X = np.concatenate([positive_data, noise_data], axis=0)
   y = np.array([1] * len(positive_data) + [0] * len(noise_data))  # 1 for P wave, 0 for noise

   dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
   dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

   train_losses = []

   ## Train the model. For now, thinking that all the type of models can take same kind of input
   if (cfg.MODEL_TYPE == MODEL_TYPE.CNN):
      print("Training CNN")
      model = PWaveCNN(cfg.SAMPLE_WINDOW_SIZE)
      criterion = nn.CrossEntropyLoss()
      optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

      for epoch in range(100):  # Example: 10 epochs.  We can increase this number to converge more
         epoch_loss = 0
         for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss+= loss.item()
         
         avg_epoch_loss = (epoch_loss/len(dataloader))
         train_losses.append(avg_epoch_loss)
         print(f'Epoch {epoch+1}, Loss: {avg_epoch_loss}')

      # Save the model
      torch.save(model.state_dict(), cfg.MODEL_FILE_NAME)
      plot_loss(train_losses)

   elif cfg.MODEL_TYPE == MODEL_TYPE.DNN:
      print("Training DNN")
      model = DNN()
      criterion = nn.CrossEntropyLoss()
      optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

      for epoch in range(100):  # Example: 10 epochs.  We can increase this number to converge more
         epoch_loss = 0 
         for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss+= loss.item()
         
         avg_epoch_loss = (epoch_loss/len(dataloader))
         train_losses.append(avg_epoch_loss)
         print(f'Epoch {epoch+1}, Loss: {avg_epoch_loss}')

      # Save the model
      torch.save(model.state_dict(), cfg.MODEL_FILE_NAME)      
      plot_loss(train_losses)


def predict(cfg): 
   print("Predicting for test set")

   model = PWaveCNN(cfg.SAMPLE_WINDOW_SIZE)
   #model = DNN()
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
 
 # Calculate the accuracy. This is tempory calculation
   true_tensor = torch.tensor(true_vrt, dtype=torch.long) 
   
   assert (predicted_classes.shape == true_tensor.shape)

   correct_predictions = (predicted_classes == true_tensor).sum().item() 
   total_predictions = true_tensor.size(0)
   accuracy = correct_predictions / total_predictions
   print(f"Prediction accuracy: {accuracy * 100:.2f}%")


def detection_accuracy(test_data, model):
   positive_count = 0
   total_data = len(test_data)
   for x in test_data:
      if predict(x) == 1:
         positive_count +=1 
   return (positive_count/total_data)*100


# IDLE method. Not useful
def idle():
   print("IN IDLE")
