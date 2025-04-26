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


# class PWaveCNN(nn.Module):
#     def __init__(self, window_size, conv1_filters=48, conv2_filters=48, dropout1=0.1 , dropout2=0.2, dropout3=0.2, fc1_neurons=24, kernel_size1=4, kernel_size2=4, model_id=""):
#         super(PWaveCNN, self).__init__()
        
#         # Convolutional layers
#         self.conv1 = nn.Conv1d(3, conv1_filters, kernel_size1)
#         self.conv2 = nn.Conv1d(conv1_filters, conv2_filters, kernel_size2)
        
#         # Pooling layers
#         #self.maxpool = nn.MaxPool1d(2)  # Max pooling with kernel size 2
#         #self.maxpool2 = nn.MaxPool1d(2)  # Max pooling with kernel size 2
#         #self.minpool = lambda x: -F.max_pool1d(-x, 2)  # Min pooling using negation trick

#         # Compute output size after convolutions and pooling
#         conv1_out_size = (window_size - kernel_size1 + 1)   # After max pool
#         conv2_out_size = (conv1_out_size - kernel_size2 + 1)  # After min pool

#         self.dropout2 = nn.Dropout(p=dropout2)
#         self.norm1 = nn.LayerNorm([conv1_filters, conv1_out_size])
#         self.norm2 = nn.GroupNorm(num_groups=2, num_channels=conv2_filters)
#         # Fully connected layers

#         self.dropout3 = nn.Dropout(p=dropout3)
#         self.fc1 = nn.Linear(conv2_filters * conv2_out_size, fc1_neurons)
#         self.fc2 = nn.Linear(fc1_neurons, 1)  # Binary classification output
        
#         # Model ID
#         random_tag = str(secrets.randbelow(9000) + 1000)
#         self.model_id = "cnn_" + datetime.now().strftime("%Y%m%d_%H%M") + "_" + random_tag if model_id == "" else model_id

#     def forward(self, x):
#         x = self.norm1(F.relu(self.conv1(x)))
#         #x = F.relu(self.conv1(x))
#         #x = self.maxpool(x)  # Apply max pooling

#         x = F.relu(self.conv2(x))
#         x = self.norm2(x)   
#         x = self.dropout2(x) 

#         #x = self.minpool(x)  # Apply min pooling

#         x = x.view(x.size(0), -1)
#         x = F.relu(self.fc1(x))
#         x = self.dropout2(x) 
#         x = F.sigmoid(self.fc2(x))
#         return x


class PWaveCNN(nn.Module):
    def __init__(self, window_size, 
                 conv1_filters=48, conv2_filters=48, conv3_filters=16,
                 dropout1=0.3 , dropout2=0.3, dropout3=0.2,
                 fc1_neurons=24, fc2_neurons=12,
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

        # self.batchnorm1 = nn.BatchNorm1d(conv1_filters)
        # self.batchnorm2 = nn.BatchNorm1d(conv2_filters)
        # self.batchnorm3 = nn.BatchNorm1d(conv3_filters)

        # Normalization and Dropout
        # self.norm1 = nn.LayerNorm([conv1_filters, conv1_out])
        # self.norm2 = nn.GroupNorm(num_groups=2, num_channels=conv2_filters)
        # self.norm3 = nn.GroupNorm(num_groups=2, num_channels=conv3_filters)

        self.dropout1 = nn.Dropout(p=dropout1)
        self.dropout2 = nn.Dropout(p=dropout2)
        #self.dropout3 = nn.Dropout(p=dropout3)

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
        #x = self.norm1(x)

        # Layer 2
        x = F.relu(self.conv2(x))
        #x = self.norm2(x)
        x = self.dropout1(x)

        # Layer 3
        x = F.relu(self.conv3(x))
        #x = self.norm3(x)
        #x = self.dropout3(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # FC Layers
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)  # reuse dropout2
        x = F.relu(self.fc2(x))
        
        x = torch.sigmoid(self.fc3(x))

        return x
