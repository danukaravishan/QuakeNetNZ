import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime

# class PWaveCNN(nn.Module):
#     def __init__(self, window_size, model_id = ""):
#         super(PWaveCNN, self).__init__()
#         self.conv1 = nn.Conv1d(3, 16, kernel_size=5)
#         self.conv2 = nn.Conv1d(16, 32, kernel_size=5)
        
#         # Calculate the correct size after convolutions
#         conv1_out_size = window_size - 5 + 1
#         conv2_out_size = conv1_out_size - 5 + 1
        
#         self.fc1 = nn.Linear(32 * conv2_out_size, 64)
#         self.fc2 = nn.Linear(64, 2)  # Binary classification: P wave or noise

#         self.model_id = "cnn_"+datetime.now().strftime("%Y%m%d_%H%M") if model_id == "" else model_id

#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = x.view(x.size(0), -1)
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x
    
## Adding one more layer
class PWaveCNN(nn.Module):
    def __init__(self, window_size, model_id=""):
        super(PWaveCNN, self).__init__()
        self.conv1 = nn.Conv1d(3, 16, kernel_size=5)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5)
        self.conv3 = nn.Conv1d(32, 32, kernel_size=5)  # New convolutional layer (32x32)
        
        # Calculate the correct size after convolutions
        conv1_out_size = window_size - 5 + 1
        conv2_out_size = conv1_out_size - 5 + 1
        conv3_out_size = conv2_out_size - 5 + 1  # Output size after the third convolution
        
        self.fc1 = nn.Linear(32 * conv3_out_size, 64)  # Update input size based on conv3 output
        self.fc2 = nn.Linear(64, 2)  # Binary classification: P wave or noise

        # Model ID with timestamp if not provided
        self.model_id = "cnn_" + datetime.now().strftime("%Y%m%d_%H%M") if model_id == "" else model_id

    def forward(self, x):
        x = F.relu(self.conv1(x))  # First convolution layer
        x = F.relu(self.conv2(x))  # Second convolution layer
        x = F.relu(self.conv3(x))  # Third convolution layer
        x = x.view(x.size(0), -1)  # Flatten the output for the fully connected layer
        x = F.relu(self.fc1(x))    # First fully connected layer
        x = self.fc2(x)            # Output layer
        return x

