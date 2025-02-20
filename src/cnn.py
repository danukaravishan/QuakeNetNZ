import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
import random

#### 1
## Adding one more layer - Original class with accuracy over 96%
# class PWaveCNN(nn.Module):
#     def __init__(self, window_size, model_id=""):
#         super(PWaveCNN, self).__init__()
#         self.conv1 = nn.Conv1d(3, 16, kernel_size=5)
#         self.conv2 = nn.Conv1d(16, 32, kernel_size=5)
#         self.conv3 = nn.Conv1d(32, 32, kernel_size=5)  # New convolutional layer (32x32)
        
#         # Calculate the correct size after convolutions
#         conv1_out_size = window_size - 5 + 1
#         conv2_out_size = conv1_out_size - 5 + 1
#         conv3_out_size = conv2_out_size - 5 + 1  # Output size after the third convolution
        
#         self.fc1 = nn.Linear(32 * conv3_out_size, 64)  # Update input size based on conv3 output
#         self.fc2 = nn.Linear(64, 2)  # Binary classification: P wave or noise

#         # Model ID with timestamp if not provided
#         self.model_id = "cnn_" + datetime.now().strftime("%Y%m%d_%H%M") if model_id == "" else model_id

#     def forward(self, x):
#         x = F.relu(self.conv1(x))  # First convolution layer
#         x = F.relu(self.conv2(x))  # Second convolution layer
#         x = F.relu(self.conv3(x))  # Third convolution layer
#         x = x.view(x.size(0), -1)  # Flatten the output for the fully connected layer
#         x = F.relu(self.fc1(x))    # First fully connected layer
#         x = self.fc2(x)            # Output layer
#         return x



###### 2 - 97% +
# class PWaveCNN(nn.Module):
#     def __init__(self, window_size, model_id=""):
#         super(PWaveCNN, self).__init__()
#         self.conv1 = nn.Conv1d(3, 16, kernel_size=5)
#         self.conv2 = nn.Conv1d(16, 16, kernel_size=5)
#         self.conv3 = nn.Conv1d(24, 24, kernel_size=5)  # New convolutional layer (32x32)
#         self.bottleneck = nn.Conv1d(24, 16, kernel_size=1)

#         # Calculate the correct size after convolutions
#         conv1_out_size = window_size - 5 + 1
#         conv2_out_size = conv1_out_size - 5 + 1
#         conv3_out_size = conv2_out_size - 5 + 1  # Output size after the third convolution
        
#         self.fc1 = nn.Linear(16 * conv2_out_size, 16)  # Update input size based on conv3 output
#         self.fc2 = nn.Linear(16, 2)  # Binary classification: P wave or noise

#         # Model ID with timestamp if not provided
#         self.model_id = "cnn_" + datetime.now().strftime("%Y%m%d_%H%M") if model_id == "" else model_id

#     def forward(self, x):
#         x = F.relu(self.conv1(x))  # First convolution layer
#         x = F.relu(self.conv2(x))  # Second convolution layer
#         x = F.relu(self.conv3(x))  # Third convolution layer
#         x = F.relu(self.bottleneck(x))
#         x = x.view(x.size(0), -1)  # Flatten the output for the fully connected layer
#         x = F.relu(self.fc1(x))    # First fully connected layer
#         x = self.fc2(x)            # Output layer
#         return x

###### 3
# class PWaveCNN(nn.Module):
#     def __init__(self, window_size, model_id=""):
#         super(PWaveCNN, self).__init__()
#         self.conv1 = nn.Conv1d(3, 16, kernel_size=5)
#         self.conv2 = nn.Conv1d(16, 16, kernel_size=5)
#         #self.conv3 = nn.Conv1d(24, 24, kernel_size=5)  # New convolutional layer (32x32)
#         #self.bottleneck = nn.Conv1d(24, 16, kernel_size=1)

#         # Calculate the correct size after convolutions
#         conv1_out_size = window_size - 5 + 1
#         conv2_out_size = conv1_out_size - 5 + 1
#         #conv3_out_size = conv2_out_size - 5 + 1  # Output size after the third convolution
          
#         self.fc1 = nn.Linear(16 * conv2_out_size, 16)  # Update input size based on conv3 output
#         self.fc2 = nn.Linear(16, 2)  # Binary classification: P wave or noise

#         # Model ID with timestamp if not provided
#         self.model_id = "cnn_" + datetime.now().strftime("%Y%m%d_%H%M") if model_id == "" else model_id

#     def forward(self, x):
#         x = F.relu(self.conv1(x))  # First convolution layer
#         x = F.relu(self.conv2(x))  # Second convolution layer
#         #x = F.relu(self.conv3(x))  # Third convolution layer
#         #x = F.relu(self.bottleneck(x))
#         x = x.view(x.size(0), -1)  # Flatten the output for the fully connected layer
#         x = F.relu(self.fc1(x))    # First fully connected layer
#         x = self.fc2(x)            # Output layer
#         return x

##### 4
# class PWaveCNN(nn.Module):
#     def __init__(self, window_size, model_id=""):
#         super(PWaveCNN, self).__init__()
#         self.conv1 = nn.Conv1d(3, 16, kernel_size=5)
#         self.conv2 = nn.Conv1d(16, 12, kernel_size=5)
#         #self.conv3 = nn.Conv1d(24, 24, kernel_size=5)  # New convolutional layer (32x32)
#         #self.bottleneck = nn.Conv1d(24, 16, kernel_size=1)

#         # Calculate the correct size after convolutions
#         conv1_out_size = window_size - 5 + 1
#         conv2_out_size = conv1_out_size - 5 + 1
#         #conv3_out_size = conv2_out_size - 5 + 1  # Output size after the third convolution
        
#         self.fc1 = nn.Linear(12 * conv2_out_size, 8)  # Update input size based on conv3 output
#         self.fc2 = nn.Linear(8, 2)  # Binary classification: P wave or noise

#         # Model ID with timestamp if not provided
#         self.model_id = "cnn_" + datetime.now().strftime("%Y%m%d_%H%M") if model_id == "" else model_id

#     def forward(self, x):
#         x = F.relu(self.conv1(x))  # First convolution layer
#         x = F.relu(self.conv2(x))  # Second convolution layer
#         #x = F.relu(self.conv3(x))  # Third convolution layer
#         #x = F.relu(self.bottleneck(x))
#         x = x.view(x.size(0), -1)  # Flatten the output for the fully connected layer
#         x = F.relu(self.fc1(x))    # First fully connected layer
#         x = self.fc2(x)            # Output layer
#         return x


## With Scalling configuration
class PWaveCNN(nn.Module):
    def __init__(self, window_size, conv1_filters=2, conv2_filters=12, fc1_neurons=8, kernel_size=3, model_id=""):
        super(PWaveCNN, self).__init__()
        
        # Convolutional layers with dynamic filters
        self.conv1 = nn.Conv1d(3, conv1_filters, kernel_size)
        self.conv2 = nn.Conv1d(conv1_filters, conv2_filters, kernel_size)
        
        # Compute output size after convolutions
        conv1_out_size = window_size - kernel_size + 1
        conv2_out_size = conv1_out_size - kernel_size + 1
        
        # Fully connected layers with dynamic neurons
        self.fc1 = nn.Linear(conv2_filters * conv2_out_size, fc1_neurons)
        self.fc2 = nn.Linear(fc1_neurons, 2)  # Binary classification output
        
        # Model ID
        random_tag = str(random.randint(1000, 9999))
        self.model_id = "cnn_" + datetime.now().strftime("%Y%m%d_%H%M") + "_" + random_tag if model_id == "" else model_id

    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    

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