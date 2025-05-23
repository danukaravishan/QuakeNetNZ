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


# CNN Larger model -  Reference
class PWaveCNNRef(nn.Module):
    def __init__(self, window_size, 
                 conv1_filters=32, conv2_filters=32, conv3_filters=32,
                 dropout1=0.3 , dropout2=0.3, dropout3=0.2,
                 fc1_neurons=44, fc2_neurons=18,
                 kernel_size1=4, kernel_size2=4, kernel_size3=4,
                 model_id=""):
        
        super(PWaveCNNRef, self).__init__()
        
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