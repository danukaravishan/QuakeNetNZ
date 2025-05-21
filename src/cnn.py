import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
import random
import numpy as np
import secrets
import pywt

# # Random seed

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# If using CUDA (GPU)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class MobileNet1D(nn.Module):
    def __init__(self, model_id=""):
        super(MobileNet1D, self).__init__()
        self.num_classes = 2
        input_channels   =3
        # Initial convolution
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        
        # Depthwise separable convolutions
        self.dw_conv1 = nn.Conv1d(32, 32, kernel_size=3, groups=32, stride=1, padding=1)
        self.pw_conv1 = nn.Conv1d(32, 64, kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm1d(64)
        
        self.dw_conv2 = nn.Conv1d(64, 64, kernel_size=3, groups=64, stride=2, padding=1)
        self.pw_conv2 = nn.Conv1d(64, 128, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm1d(128)

        # Global average pooling and classification
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # Reduces time dimension to 1
        self.fc = nn.Linear(128, self.num_classes)
        self.model_id = "mobilenet_" + datetime.now().strftime("%Y%m%d_%H%M") if model_id == "" else model_id

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.pw_conv1(self.dw_conv1(x))))
        x = F.relu(self.bn3(self.pw_conv2(self.dw_conv2(x))))
        x = self.global_pool(x)  # Shape: (batch_size, 128, 1)
        x = torch.flatten(x, 1)  # Shape: (batch_size, 128)
        x = self.fc(x)  # Shape: (batch_size, num_classes)
        return x


# CNN Larger model -  Reference
class PWaveCNN(nn.Module):
    def __init__(self, window_size, 
                 conv1_filters=32, conv2_filters=32, conv3_filters=32,
                 dropout1=0.3 , dropout2=0.3, dropout3=0.2,
                 fc1_neurons=44, fc2_neurons=18,
                 kernel_size1=4, kernel_size2=4, kernel_size3=4,
                 model_id=""):
        
        super(PWaveCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv1d(3, conv1_filters, kernel_size1)
        self.conv2 = nn.Conv1d(conv1_filters, conv2_filters, kernel_size2)
        self.conv3 = nn.Conv1d(conv2_filters, conv3_filters, kernel_size3)

        # Compute output sizes
        conv1_out = window_size - kernel_size1 + 1
        conv2_out = conv1_out - kernel_size2 + 1
        conv3_out = conv2_out - kernel_size3 + 1

        self.dropout1 = nn.Dropout(p=dropout1)
        self.dropout2 = nn.Dropout(p=dropout2)

        # Fully connected layers
        self.fc1 = nn.Linear(conv3_filters * conv3_out, fc1_neurons)
        self.fc2 = nn.Linear(fc1_neurons, fc2_neurons)
        self.fc3 = nn.Linear(fc2_neurons, 1)  # Binary classification

        # Model ID
        random_tag = str(secrets.randbelow(9000) + 1000)
        self.model_id = "cnn_" + datetime.now().strftime("%Y%m%d_%H%M") + "_" + random_tag if model_id == "" else model_id

    def forward(self, x):
        # Layer 1
        x = F.relu(self.conv1(x))

        # Layer 2
        x = F.relu(self.conv2(x))
        x = self.dropout1(x)

        # Layer 3
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)

        # FC Layers
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)  # reuse dropout2
        x = F.relu(self.fc2(x))
        
        x = torch.sigmoid(self.fc3(x))

        return x
    
# Experiment Model - Last experiment - Depthwise seperable layers at the front
# class PWaveCNN(nn.Module):
#     def __init__(self, window_size, 
#                  conv1_filters=32, conv2_filters=32, conv3_filters=32,
#                  dropout1=0.3 , dropout2=0.3, dropout3=0.2,
#                  fc1_neurons=44, fc2_neurons=18,
#                  kernel_size1=4, kernel_size2=4, kernel_size3=4,
#                  model_id=""):
        
#         super(PWaveCNN, self).__init__()
        
#         # Convolutional layers
#         # Replace the first two layers with depthwise separable convolutions
#         self.depthwise_conv1 = nn.Conv1d(3, 3, kernel_size1, groups=3)  # Depthwise convolution for first layer
#         self.pointwise_conv1 = nn.Conv1d(3, conv1_filters, kernel_size=1)  # Pointwise convolution for first layer

#         self.depthwise_conv2 = nn.Conv1d(conv1_filters, conv1_filters, kernel_size2, groups=conv1_filters)  # Depthwise convolution for second layer
#         self.pointwise_conv2 = nn.Conv1d(conv1_filters, conv2_filters, kernel_size=1)  # Pointwise convolution for second layer

#         self.conv3 = nn.Conv1d(conv2_filters, conv3_filters, kernel_size3)

#         # Compute output sizes
#         conv1_out = window_size - kernel_size1 + 1
#         conv2_out = conv1_out - kernel_size2 + 1
#         conv3_out = conv2_out - kernel_size3 + 1

#         self.dropout1 = nn.Dropout(p=dropout1)
#         self.dropout2 = nn.Dropout(p=dropout2)

#         # Fully connected layers
#         self.fc1 = nn.Linear(conv3_filters * conv3_out, fc1_neurons)
#         self.fc2 = nn.Linear(fc1_neurons, fc2_neurons)
#         self.fc3 = nn.Linear(fc2_neurons, 1)  # Binary classification

#         # Model ID
#         random_tag = str(secrets.randbelow(9000) + 1000)
#         self.model_id = "cnn_" + datetime.now().strftime("%Y%m%d_%H%M") + "_" + random_tag if model_id == "" else model_id

#     def forward(self, x):
#         # Layer 1
#         x = F.relu(self.depthwise_conv1(x))
#         x = F.relu(self.pointwise_conv1(x))

#         # Layer 2
#         x = F.relu(self.depthwise_conv2(x))
#         x = F.relu(self.pointwise_conv2(x))
#         x = self.dropout1(x)

#         # Layer 3
#         x = F.relu(self.conv3(x))
#         x = x.view(x.size(0), -1)

#         # FC Layers
#         x = F.relu(self.fc1(x))
#         x = self.dropout2(x)  # reuse dropout2
#         x = F.relu(self.fc2(x))
        
#         x = torch.sigmoid(self.fc3(x))

#         return x





class PDetector(nn.Module):
    def __init__(self, 
                 n_fft=32, hop_length=64, window_fn=torch.hann_window,
                 lstm_hidden=32, lstm_layers=1, 
                 dropout_cnn=0.2, dropout_lstm=0.2,
                 fc_neurons=64, model_id=""):
        super(PDetector, self).__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window_fn = window_fn

        # CNN
        self.conv1 = nn.Conv2d(3, 8, kernel_size=(5,2), stride=2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(3,1), stride=2)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=(3,1), stride=1)

        self.dropout_cnn = nn.Dropout(dropout_cnn)
        self.bn1 = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(16)
        self.bn3 = nn.BatchNorm2d(16)

        # LSTM setup — we will create these in `forward()` after seeing input shape
        self.lstm = None
        self.lstm2 = None

        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        self.dropout_lstm = nn.Dropout(dropout_lstm)

        self.fc1 = None
        self.fc2 = None
        self.fc_neurons = fc_neurons

        random_tag = str(secrets.randbelow(9000) + 1000)
        self.model_id = "pdetectorstft_" + datetime.now().strftime("%Y%m%d_%H%M") + "_" + random_tag if model_id == "" else model_id

    def compute_stft(self, x):
        B, C, T = x.shape
        window = self.window_fn(self.n_fft).to(x.device)

        stfts = []
        for ch in range(C):
            stft = torch.stft(x[:, ch], 
                              n_fft=self.n_fft, 
                              hop_length=self.hop_length, 
                              window=window, 
                              return_complex=True)
            mag = torch.abs(stft)
            stfts.append(mag)
        spectrogram = torch.stack(stfts, dim=1)  # (B, C, F, T)
        return spectrogram

    def forward(self, x):
        # x: (B, 3, T)
        x = self.compute_stft(x)  # -> (B, 3, F, T)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout_cnn(x)

        b, c, h, w = x.shape
        x = x.permute(0, 2, 1, 3).contiguous().view(b, h, c * w)  # (B, seq_len, feature_size)

        # Create LSTM and FC layers dynamically based on feature_size
        if self.lstm is None:
            self.lstm_input_size = x.size(-1)
            self.lstm = nn.LSTM(input_size=self.lstm_input_size,
                                hidden_size=self.lstm_hidden,
                                num_layers=self.lstm_layers,
                                batch_first=True,
                                bidirectional=True).to(x.device)
            
            self.lstm2 = nn.LSTM(input_size=2 * self.lstm_hidden,
                                 hidden_size=self.lstm_hidden,
                                 num_layers=1,
                                 batch_first=True).to(x.device)

            self.fc1 = nn.Linear(self.lstm_hidden, self.fc_neurons).to(x.device)
            self.fc2 = nn.Linear(self.fc_neurons, 1).to(x.device)

        x, _ = self.lstm(x)
        x, _ = self.lstm2(x)
        x = x[:, -1, :]  # last time step

        x = self.dropout_lstm(x)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x