from utils import *
from database_op import *
from config import Config, MODE_TYPE, MODEL_TYPE, NNCFG
import optuna
from torch.utils.data import random_split, DataLoader, TensorDataset

def _train(model, dataloader, optimizer, criterion, epoch_iter=50):

   train_losses = []
   for epoch in range(epoch_iter):
      epoch_loss = 0
      for batch_X, batch_y in dataloader:
         optimizer.zero_grad()
         output = model(batch_X)
         loss = criterion(output.squeeze(), batch_y)
         loss.backward()
         optimizer.step()
         epoch_loss+= loss.item()
      
      avg_epoch_loss = (epoch_loss/len(dataloader))
      train_losses.append(avg_epoch_loss)
      print(f'Epoch {epoch+1}, Loss: {avg_epoch_loss}')

   return model,train_losses


def train(cfg):
   
   nncfg = NNCFG()
   nncfg.argParser()

   hdf5_file = h5py.File(cfg.TRAIN_DATA, 'r')
   p_data, s_data, noise_data = getWaveData(cfg, hdf5_file)
   
   # Data preparation
   p_data = np.array(p_data)
   s_data = np.array(s_data)
   noise_data = np.array(noise_data)

   positive_data = np.concatenate((p_data , s_data))

   X = np.concatenate([positive_data, noise_data], axis=0)
   Y = np.array([1] * len(positive_data) + [0] * len(noise_data))  # 1 for P wave, 0 for noise

   dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.long))
   dataloader = DataLoader(dataset, batch_size=nncfg.batch_size, shuffle=True)

   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   model = None

   ## Train the model. For now, thinking that all the type of models can take same kind of input
   if (cfg.MODEL_TYPE == MODEL_TYPE.CNN):
      model = PWaveCNN(cfg.SAMPLE_WINDOW_SIZE).to(device)
      criterion = nn.CrossEntropyLoss()
      optimizer = torch.optim.Adam(model.parameters(), lr=nncfg.learning_rate)
      model, train_losses = _train(model, dataloader, optimizer, criterion, nncfg.epoch_count)

   elif cfg.MODEL_TYPE == MODEL_TYPE.DNN:
      model = DNN().to(device)
      model.apply(InitWeights)
      #criterion = nn.CrossEntropyLoss()
      optimizer = torch.optim.Adam(model.parameters(), lr=nncfg.learning_rate)
      criterion = nn.BCEWithLogitsLoss()
      model, train_losses = _train(model, dataloader, optimizer, criterion, nncfg.epoch_count)

   elif cfg.MODEL_TYPE == MODEL_TYPE.PhaseNet:
         model = sbm.PhaseNet(phases="PSN", norm="peak")
         #criterion = nn.CrossEntropyLoss()
         optimizer = torch.optim.Adam(model.parameters(), lr=nncfg.learning_rate)
         criterion = nn.BCEWithLogitsLoss()
         model, train_losses = _train(model, dataloader, optimizer, criterion, nncfg.epoch_count)

   elif cfg.MODEL_TYPE == MODEL_TYPE.CRED:
        # Define the CRED model
        input_shape = (X.shape[1], X.shape[2], 1)  # assuming X is formatted as (batch, height, width, channels)
        filters = [32, 64, 128, 256]  # example filter values, adjust as needed
        model = CRED(input_shape=input_shape, filters=filters).to(device)
        criterion = nn.BCEWithLogitsLoss()  # Use Binary Cross-Entropy as we're doing binary classification
        optimizer = torch.optim.Adam(model.parameters(), lr=nncfg.learning_rate)
        model, train_losses = _train(model, dataloader, optimizer, criterion, nncfg.epoch_count)
   
   cfg.MODEL_FILE_NAME = cfg.MODEL_PATH + model.model_id

   # Save the model
   torch.save({
      'model_state_dict': model.state_dict(),
      'model_id'        : model.model_id,  # Save model ID
      'epoch_count'     : nncfg.epoch_count,
      'learning_rate'   : nncfg.learning_rate,
      'batch_size'      : nncfg.batch_size,
      'optimizer'       : optimizer.__class__.__name__.lower(),
      'training_loss'   : train_losses
   }, cfg.MODEL_FILE_NAME + ".pt")

   plot_loss(train_losses, cfg.MODEL_FILE_NAME)
   cfg.MODEL_FILE_NAME += ".pt"


def objective(trial):
   nncfg = NNCFG()
   nncfg.argParser()
   cfg = Config()

   hdf5_file = h5py.File(cfg.TRAIN_DATA, 'r')
   p_data, s_data, noise_data = getWaveData(cfg, hdf5_file)
   
   # Data preparation
   p_data = np.array(p_data)
   s_data = np.array(s_data)
   noise_data = np.array(noise_data)

   positive_data = np.concatenate((p_data , s_data))

   X = np.concatenate([positive_data, noise_data], axis=0)
   Y = np.array([1] * len(positive_data) + [0] * len(noise_data))  # 1 for P wave, 0 for noise

   dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.long))

   train_ratio = 0.8  # 80% for training
   val_ratio = 0.2    # 20% for validation
   train_size = int(train_ratio * len(dataset))
   val_size = len(dataset) - train_size

   train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   model = None

   ## Train the model. For now, thinking that all the type of models can take same kind of input
   # if (cfg.MODEL_TYPE == MODEL_TYPE.CNN):
   #    model = PWaveCNN(cfg.SAMPLE_WINDOW_SIZE).to(device)
   #    criterion = nn.CrossEntropyLoss()
   #    optimizer = torch.optim.Adam(model.parameters(), lr=nncfg.learning_rate)
   #    model, train_losses = _train(model, dataloader, optimizer, criterion, nncfg.epoch_count)

    # Define hyperparameters to tune
   lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)  # Learning rate
   batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])  # Batch size
   conv1_filters = trial.suggest_int('conv1_filters', 8, 64, step=8)  # Conv1 filters
   conv2_filters = trial.suggest_int('conv2_filters', 16, 128, step=16)  # Conv2 filters

    # Create DataLoader with the suggested batch size
   train_loader = DataLoader(train_dataset, batch_size=nncfg.batch_size, shuffle=True)
   val_loader = DataLoader(val_dataset, batch_size=nncfg.batch_size, shuffle=False)

    # Define the model
   # class TunablePWaveCNN(PWaveCNN):
   #    def __init__(self, window_size, conv1_filters, conv2_filters):
   #       super().__init__(window_size)
   #       self.conv1 = nn.Conv1d(3, conv1_filters, kernel_size=5)
   #       self.conv2 = nn.Conv1d(conv1_filters, conv2_filters, kernel_size=5)
   #       conv1_out_size = window_size - 5 + 1
   #       conv2_out_size = conv1_out_size - 5 + 1
   #       self.fc1 = nn.Linear(conv2_filters * conv2_out_size, 64)

   model = PWaveCNN(cfg.SAMPLE_WINDOW_SIZE).to(device)
   #window_size = 200  # Replace with your actual input size
   #model = TunablePWaveCNN(window_size, conv1_filters, conv2_filters)
   #model = model.to(device)  # Send model to GPU if available

   # Define optimizer and loss function
   optimizer = torch.optim.Adam(model.parameters(), lr=lr)
   criterion = nn.CrossEntropyLoss()

   # Training
   _train(model, train_loader, optimizer, criterion, epoch_iter=2)  # Use fewer epochs for faster tuning

   # Validation
   model.eval()
   val_loss = 0
   with torch.no_grad():
      for batch_X, batch_y in val_loader:
         batch_X, batch_y = batch_X.to(device), batch_y.to(device)
         output = model(batch_X)
         loss = criterion(output.squeeze(), batch_y)
         val_loss += loss.item()

   avg_val_loss = val_loss / len(val_loader)
   return avg_val_loss  # Optuna minimizes this value

# Run Optuna Study

def hyper_param_opt():
   study = optuna.create_study(direction='minimize')
   study.optimize(objective, n_trials=50)
   # Best hyperparameters
   print("Best hyperparameters: ", study.best_params)