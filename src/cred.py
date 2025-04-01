## This is a copy of CRED model to be trained with shorter input window
## This model is still not being used.
## Seisbench CRED is only being used upto now

import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime


class BlockCNN(nn.Module):
    def __init__(self, filters, kernel_size):
        super(BlockCNN, self).__init__()
        self.bn1 = nn.BatchNorm2d(filters)
        self.conv1 = nn.Conv2d(filters, filters, kernel_size=kernel_size-2, padding=(kernel_size-2)//2)
        self.bn2 = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=kernel_size-2, padding=(kernel_size-2)//2)

    def forward(self, x):
        x = F.relu(self.bn1(x))
        x = self.conv1(x)
        x = F.relu(self.bn2(x))
        x = self.conv2(x)
        return x

class BlockBiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, rnn_depth):
        super(BlockBiLSTM, self).__init__()
        self.rnn_depth = rnn_depth
        self.lstm_layers = nn.ModuleList([
            nn.LSTM(input_dim if i == 0 else hidden_dim * 2, hidden_dim, batch_first=True, bidirectional=True)
            for i in range(rnn_depth)
        ])
        self.dropout = nn.Dropout(0.7)

    def forward(self, x):
        residual = x
        for i, lstm in enumerate(self.lstm_layers):
            x, _ = lstm(x)
            x = self.dropout(x)
            if i > 0:
                x = x + residual  # Residual connection
            residual = x
        return x

class CRED(nn.Module):
    def __init__(self, input_shape, filters, model_id=""):
        super(CRED, self).__init__()
        
        self.model_id = "cred_" + datetime.now().strftime("%Y%m%d_%H%M") if model_id == "" else model_id
        # First Conv2D layer
        self.conv2D_2 = nn.Conv2d(in_channels=input_shape[0], out_channels=filters[0], kernel_size=9, stride=2, padding=4)
        self.res_conv_2 = BlockCNN(filters[0], 9)
        
        # Second Conv2D layer
        self.conv2D_3 = nn.Conv2d(in_channels=filters[0], out_channels=filters[1], kernel_size=5, stride=2, padding=2)
        self.res_conv_3 = BlockCNN(filters[1], 5)
        
        # BiLSTM and UniLSTM Layers
        self.bilstm = BlockBiLSTM(filters[1] * input_shape[1] // 4, filters[3], rnn_depth=2)
        self.unilstm = nn.LSTM(filters[3] * 2, filters[3], batch_first=True)
        self.dropout1 = nn.Dropout(0.8)
        self.bn1 = nn.BatchNorm1d(filters[3])

        # Dense Layers
        self.dense_2 = nn.Linear(filters[3], filters[3])
        self.bn2 = nn.BatchNorm1d(filters[3])
        self.dropout2 = nn.Dropout(0.8)
        self.output_dense = nn.Linear(filters[3], 1)

    def forward(self, x):
        # First Conv2D + Residual Block
        x = F.relu(self.conv2D_2(x))
        x = self.res_conv_2(x) + x  # Residual connection for block_CNN
        
        # Second Conv2D + Residual Block
        x = F.relu(self.conv2D_3(x))
        x = self.res_conv_3(x) + x  # Residual connection for block_CNN
        
        # Reshape layer
        batch_size, channels, height, width = x.size()
        x = x.view(batch_size, height, -1)  # Reshape to (batch_size, height, channels * width)
        
        # BiLSTM Block
        x = self.bilstm(x)
        
        # UniLSTM Block
        x, _ = self.unilstm(x)
        x = self.dropout1(x)
        
        # TimeDistributed Dense Layers
        x = x.contiguous().view(-1, x.size(-1))  # Flatten for BatchNorm and Dense layers
        x = self.bn1(x)
        x = F.relu(self.dense_2(x))
        x = self.bn2(x)
        x = self.dropout2(x)
        x = torch.sigmoid(self.output_dense(x))
        
        # Reshape back to time-distributed format
        x = x.view(batch_size, height, -1)
        return x

def lr_schedule(epoch):
    """
    Learning rate scheduler for PyTorch.
    """
    lr = 1e-3
    if epoch > 60:
        lr *= 0.5e-3
    elif epoch > 40:
        lr *= 1e-3
    elif epoch > 20:
        lr *= 1e-2
    elif epoch > 10:
        lr *= 1e-1
    print(f'Learning rate: {lr}')
    return lr
