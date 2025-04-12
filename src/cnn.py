import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
import random
import numpy as np
import secrets

# Random seed

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


class PWaveCNN(nn.Module):
    def __init__(self, window_size, conv1_filters=24, conv2_filters=4, dropout1=0.1 , dropout2=0.1, dropout3=0.1, fc1_neurons=20, kernel_size1=4, kernel_size2=4, model_id=""):
        
        # Base
        # conv1_filters=8
        # conv2_filters=2
        # fc1_neurons=22
        # kernel_size1=4
        # kernel_size2=4

        super(PWaveCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv1d(3, conv1_filters, kernel_size1)
        self.conv2 = nn.Conv1d(conv1_filters, conv2_filters, kernel_size2)
        
        # Pooling layers
        #self.maxpool = nn.MaxPool1d(2)  # Max pooling with kernel size 2
        #self.maxpool2 = nn.MaxPool1d(2)  # Max pooling with kernel size 2
        #self.minpool = lambda x: -F.max_pool1d(-x, 2)  # Min pooling using negation trick

        # Compute output size after convolutions and pooling
        conv1_out_size = (window_size - kernel_size1 + 1)   # After max pool
        conv2_out_size = (conv1_out_size - kernel_size2 + 1)  # After min pool
        
        # Fully connected layers
        self.fc1 = nn.Linear(conv2_filters * conv2_out_size, fc1_neurons)
        self.fc2 = nn.Linear(fc1_neurons, 1)  # Binary classification output
        
        # Model ID
        random_tag = str(secrets.randbelow(9000) + 1000)
        self.model_id = "cnn_" + datetime.now().strftime("%Y%m%d_%H%M") + "_" + random_tag if model_id == "" else model_id

    def forward(self, x):
        x = F.relu(self.conv1(x))
        #x = self.maxpool(x)  # Apply max pooling

        x = F.relu(self.conv2(x))
        #x = self.minpool(x)  # Apply min pooling

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        return x
