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
   dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

   train_losses = []
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

   ## Train the model. For now, thinking that all the type of models can take same kind of input
   if (cfg.MODEL_TYPE == MODEL_TYPE.CNN):
      print("Training CNN")
      model = PWaveCNN(cfg.SAMPLE_WINDOW_SIZE).to(device)
      model.load_state_dict(torch.load(cfg.MODEL_FILE_NAME))
      model.eval()
      criterion = nn.CrossEntropyLoss()
      optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

      for epoch in range(50):  # Example: 10 epochs.  We can increase this number to converge more
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
      model = DNN().to(device)
      model.apply(InitWeights)
      #criterion = nn.CrossEntropyLoss()
      optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
      scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
      #criterion = nn.BCELoss()
      criterion = nn.BCEWithLogitsLoss()

      for epoch in range(50):  # Example: 10 epochs.  We can increase this number to converge more
         epoch_loss = 0 
         for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            output = model(batch_X).squeeze()
            batch_y = batch_y.float()
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss+= loss.item()
         
         scheduler.step()
         avg_epoch_loss = (epoch_loss/len(dataloader))
         train_losses.append(avg_epoch_loss)
         print(f'Epoch {epoch+1}, Loss: {avg_epoch_loss}')

      # Save the model
      torch.save(model.state_dict(), cfg.MODEL_FILE_NAME)      
      plot_loss(train_losses)

