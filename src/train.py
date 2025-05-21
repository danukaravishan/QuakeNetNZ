
from utils import *
from database_op import *
from config import Config, MODE_TYPE, MODEL_TYPE, NNCFG
import optuna
from torch.utils.data import random_split, DataLoader, TensorDataset
from dataprep import pre_proc_data,normalize, normalize_data, apply_wavelet_denoise
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from report  import plot_loss


def _train(model, dataloader, val_loader, optimizer, criterion, epoch_iter=50):

   device = next(model.parameters()).device
   train_losses = []
   val_losses = []
   val_accuracies = []

   for epoch in range(epoch_iter):
      model.train() 
      epoch_loss = 0

      for batch_X, batch_y in dataloader:
         batch_X, batch_y = batch_X.to(device), batch_y.to(device)
         optimizer.zero_grad()
         output = model(batch_X)
         #loss = criterion(output.squeeze(), batch_y)
         loss = criterion(output.view(-1), batch_y)
         loss.backward()
         optimizer.step()
         epoch_loss+= loss.item()

      avg_epoch_loss = (epoch_loss/len(dataloader))
      train_losses.append(avg_epoch_loss)
      print(f'Epoch {epoch+1}, Loss: {avg_epoch_loss:.5f}', end="")      
      
      model.eval()  # Set model to evaluation mode
      val_loss = 0
      correct = 0
      total = 0

      with torch.no_grad():
         for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            outputs = model(batch_X)  # Shape: [batch_size, num_classes]

            # Ensure batch_y has correct shape
            if len(batch_y.shape) > 1:  # If batch_y is one-hot encoded
                  batch_y = batch_y.argmax(dim=1)
            else:
                  batch_y = batch_y.view(-1)  # Ensures it's 1D

            loss = criterion(outputs.squeeze(), batch_y)
            val_loss += loss.item()

            #_, predicted = torch.max(outputs, 1)  # Get predicted class index

            predicted = (outputs > 0.5)
            predicted = predicted.squeeze()
            total += batch_y.size(0)  
            correct += (predicted == batch_y).sum().item()

      val_losses.append(val_loss / len(val_loader))
      val_accuracies.append(100 * correct / total)
      print(f'  Val_loss: {(val_loss / len(val_loader)):.5f},  Validation Accuracy : {(100 * correct / total):.5f}')  

   return model, train_losses, val_losses, val_accuracies 


def train(cfg):
   
   nncfg = NNCFG()
   nncfg.argParser()

   hdf5_file = h5py.File(cfg.TRAIN_DATA, 'r')

   p_data, s_data, noise_data = getWaveData(cfg, hdf5_file)
    
   # Data preparation
   p_data = np.array(p_data)
   s_data = np.array(s_data)
   noise_data = np.array(noise_data)

   p_data = normalize_data(p_data)
   s_data = normalize_data(s_data)
   noise_data = normalize_data(noise_data)

   ## Merge First 20% of test data into validation set
   hdf5_file_test = h5py.File(cfg.TEST_DATA, 'r')
   p_data_test, s_data_test, noise_data_test = getWaveData(cfg, hdf5_file_test)

   p_data_test = np.array(p_data_test)
   s_data_test = np.array(s_data_test)
   noise_data_test = np.array(noise_data_test)

   test_val_split_ratio = 0.5
   
   random_state = np.random.RandomState(42)  # Set a fixed random seed for consistency
   random_indices = random_state.choice(len(p_data_test), int(test_val_split_ratio * p_data_test.shape[0]), replace=False)
   p_data_test = p_data_test[random_indices]

   random_indices = random_state.choice(len(s_data_test), int(test_val_split_ratio * s_data_test.shape[0]), replace=False)
   s_data_test = s_data_test[random_indices]

   random_indices = random_state.choice(len(noise_data_test), int(test_val_split_ratio * noise_data_test.shape[0]), replace=False)
   noise_data_test = noise_data_test[random_indices]

   p_data_test = normalize_data(p_data_test)
   s_data_test = normalize_data(s_data_test)
   noise_data_test = normalize_data(noise_data_test)
   
   ### 
   positive_data = np.concatenate((p_data , s_data))
   X = np.concatenate([positive_data, noise_data], axis=0)
   Y = np.array([1] * len(positive_data) + [0] * len(noise_data))  # 1 for P wave, 0 for noise

   full_dataset = list(zip(X, Y))
   random.Random(42).shuffle(full_dataset)
   X, Y = zip(*full_dataset)
   X = np.array(X)
   Y = np.array(Y)

   train_size = int(1 * len(X))

   X_train = X[:train_size]
   Y_train = Y[:train_size]

   X_val = X[train_size:]
   Y_val = Y[train_size:]

   X_test_val = np.concatenate([p_data_test, s_data_test, noise_data_test])
   Y_test_val = np.array([1] * (len(p_data_test) + len(s_data_test)) + [0] * len(noise_data_test))

   X_val = np.concatenate([X_val, X_test_val])
   Y_val = np.concatenate([Y_val, Y_test_val])

   train_loader = DataLoader(
    TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                  torch.tensor(Y_train, dtype=torch.float)),
    batch_size=nncfg.batch_size,
    shuffle=True
   )
   
   val_loader = DataLoader(
    TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                  torch.tensor(Y_val, dtype=torch.float)),
    batch_size=nncfg.batch_size,
    shuffle=False
   )
 
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   model = None

   # Remove random initialisation

   ## Train the model. For now, thinking that all the type of models can take same kind of input
   if (cfg.MODEL_TYPE == MODEL_TYPE.CNN):
      model = PWaveCNN( 
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
      ).to(device)

      #criterion = nn.CrossEntropyLoss()
      criterion = nn.BCELoss()
      optimizer = torch.optim.Adam(model.parameters(), lr=nncfg.learning_rate, weight_decay=nncfg.l2_decay)

      model, train_losses, val_loss, val_acc = _train(model, train_loader, val_loader, optimizer, criterion, nncfg.epoch_count)

   elif (cfg.MODEL_TYPE == MODEL_TYPE.MobileNet1D):
      model = MobileNet1D().to(device)
      criterion = nn.CrossEntropyLoss()
      optimizer = torch.optim.Adam(model.parameters(), lr=nncfg.learning_rate)
      model, train_losses = _train(model, train_loader, optimizer, criterion, nncfg.epoch_count)

   elif cfg.MODEL_TYPE == MODEL_TYPE.PDetector:
      model = PDetector().to(device)
      criterion = nn.BCELoss()
      optimizer = torch.optim.Adam(model.parameters())
      model, train_losses, val_loss, val_acc = _train(model, train_loader, val_loader, optimizer, criterion, nncfg.epoch_count)
   
   elif cfg.MODEL_TYPE == MODEL_TYPE.PhaseNet:
         model = sbm.PhaseNet(phases="PSN", norm="peak")
         #criterion = nn.CrossEntropyLoss()
         optimizer = torch.optim.Adam(model.parameters(), lr=nncfg.learning_rate)
         criterion = nn.BCEWithLogitsLoss()
         model, train_losses = _train(model, train_loader, optimizer, criterion, nncfg.epoch_count)

   elif cfg.MODEL_TYPE == MODEL_TYPE.CRED:
        # Define the CRED model
        input_shape = (X.shape[1], X.shape[2], 1)  # assuming X is formatted as (batch, height, width, channels)
        filters = [32, 64, 128, 256]  # example filter values, adjust as needed
        model = CRED(input_shape=input_shape, filters=filters).to(device)
        criterion = nn.BCEWithLogitsLoss()  # Use Binary Cross-Entropy as we're doing binary classification
        optimizer = torch.optim.Adam(model.parameters(), lr=nncfg.learning_rate)
        model, train_losses = _train(model, train_loader, optimizer, criterion, nncfg.epoch_count)
   
   cfg.MODEL_FILE_NAME = cfg.MODEL_PATH + model.model_id

   # Save the model
   # torch.save({
   #    'model_state_dict': model.state_dict(),
   #    'model_id'        : model.model_id,  # Save model ID
   #    'epoch_count'     : nncfg.epoch_count,
   #    'learning_rate'   : nncfg.learning_rate,
   #    'batch_size'      : nncfg.batch_size,
   #    'optimizer'       : optimizer.__class__.__name__.lower(),
   #    'training_loss'   : train_losses,
   #    'validation_acc' :  val_acc
   # }, cfg.MODEL_FILE_NAME + ".pt")

   sample = torch.tensor(X_train[0], dtype=torch.float32).unsqueeze(0)
   sample = sample.to(device)
   traced_model = torch.jit.trace(model, sample)
   torch.jit.save(traced_model, cfg.MODEL_FILE_NAME+".pt")

   plot_loss(train_losses, val_loss,  val_acc, cfg.MODEL_FILE_NAME)
   cfg.MODEL_FILE_NAME += ".pt"

