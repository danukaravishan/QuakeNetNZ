

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

def _train_tfeq(model, dataloader, val_loader, optimizer, criterion, epoch_iter=50):

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
         loss = F.nll_loss(output, batch_y.squeeze())
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

            #loss = criterion(outputs.squeeze(), batch_y)
            val_loss += F.nll_loss(outputs, batch_y.squeeze()).item()

            pred = outputs.max(1, keepdim=True)[1]
            correct += pred.eq(batch_y.view_as(pred)).sum().item()
            total += batch_y.size(0)
            
      val_losses.append(val_loss / len(val_loader))
      val_accuracies.append(100 * correct / total)
      print(f'  Val_loss: {(val_loss / len(val_loader)):.5f},  Validation Accuracy : {(100 * correct / total):.5f}')  

   return model, train_losses, val_losses, val_accuracies 