from utils import *
from database_op import *
from config import Config, MODE_TYPE, MODEL_TYPE, NNCFG
from unet import uNet

def _train(model, dataloader, optimizer, criterion, threshold= 0.4, epoch_iter=50):

   train_losses = []
   for epoch in range(epoch_iter):
      epoch_loss = 0
      for batch_X, batch_y in dataloader:
         # print(f"Batch X shape: {batch_X.shape}, Batch Y shape: {batch_y.shape}")
         optimizer.zero_grad()
         output = model(batch_X) # 32, 3 ,200
         #assert output.requires_grad, "Model output does not require gradients!"
         #print("threshold:", threshold)
         #print("Output Max:", output[:,1].max().item())
         # Print model output values
         #print("Output values:", output)
         #print("Output shape:", output.shape)
         #print("Output data type:", output.dtype)
         # output = torch.mean(output, dim=2)
         # Thresholding the output
         # print("Output [1] values:", output[:,1])
         # exceeds_threshold = output[:, 1] > threshold
         # print("Exceeds threshold:", exceeds_threshold)
         p_exceeds = torch.any(output[:, 1] > threshold, dim=1)
         # print("P exceeds:", p_exceeds)
         # print("Output [2] values:", output[:,2])
         s_exceeds = torch.any(output[:, 2] > threshold, dim=1)
         # print("S exceeds:", s_exceeds)
         outputs = torch.where(torch.logical_or(p_exceeds, s_exceeds), 
                              torch.tensor(1.0, device = output.device), 
                              torch.tensor(0.0, device = output.device))
         # outputs = torch.logical_or(p_exceeds, s_exceeds).float()
         # outputs = (output[:, 1] > threshold) | (output[:, 2] > threshold)
         # outputs = outputs.float()
         # Debug: Check output and target shapes
         #print(f"Outputs shape: {outputs.shape}, Target shape: {batch_y.shape}")
         # Print model output values
         #print("Output values:", outputs)
         #print("Output shape:", outputs.shape)
         #print("Output data type:", outputs.dtype)
         # loss = criterion(output.squeeze(), batch_y)
         batch_y= batch_y.float()
         #print("Batch Y values:", batch_y)
         #print("Batch Y shape:", batch_y.shape)
         #print("Batch Y data type:", batch_y.dtype)
         loss = criterion(outputs, batch_y)
         loss.requires_grad_(True)
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

   # print(cfg.TRAIN_DATA)
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
   
   elif cfg.MODEL_TYPE == MODEL_TYPE.UNET:
      model = uNet(in_channels=3, out_channels = 3).to(device)
      # As they have used cross-entropy in the paper, I am using the same
      # criterion = nn.CrossEntropyLoss()
      criterion = nn.BCEWithLogitsLoss()
      optimizer = torch.optim.Adam(model.parameters(), lr=nncfg.learning_rate)
      model, train_losses = _train(model, dataloader, optimizer, criterion, epoch_iter=nncfg.epoch_count)

   
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
