from utils import *
from database_op import *
from config import Config, MODE_TYPE, MODEL_TYPE

def _train(model, dataloader, optimizer, criterion, epoch_iter):

   train_losses = []
   for epoch in range(epoch_iter):
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

   return model,train_losses


def train(cfg):
      
   hdf5_file = h5py.File("data/train_data", 'r')
   p_data, s_data, noise_data = getWaveData(cfg, hdf5_file)
   
   # Data preparation
   p_data = np.array(p_data)
   s_data = np.array(s_data)
   noise_data = np.array(noise_data)

   positive_data = np.concatenate((p_data , s_data))

   X = np.concatenate([positive_data, noise_data], axis=0)
   Y = np.array([1] * len(positive_data) + [0] * len(noise_data))  # 1 for P wave, 0 for noise

   dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.long))
   dataloader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)

   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


   ## Train the model. For now, thinking that all the type of models can take same kind of input
   if (cfg.MODEL_TYPE == MODEL_TYPE.CNN):
      model = PWaveCNN(cfg.SAMPLE_WINDOW_SIZE).to(device)
      criterion = nn.CrossEntropyLoss()
      optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
      model, train_losses = _train(model, dataloader, optimizer, criterion, 10)

      # Save the model
      torch.save(model.state_dict(), cfg.MODEL_FILE_NAME)
      plot_loss(train_losses)


   elif cfg.MODEL_TYPE == MODEL_TYPE.DNN:
      model = DNN().to(device)
      model.apply(InitWeights)
      #criterion = nn.CrossEntropyLoss()
      optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
      criterion = nn.BCEWithLogitsLoss()
      model, train_losses = _train(model, dataloader, optimizer, criterion, 50)

      # Save the model
      torch.save(model.state_dict(), cfg.MODEL_FILE_NAME)      
      plot_loss(train_losses)

