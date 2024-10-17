import torch
import torch.nn as nn
import torch.nn.functional as F

class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.input_layer = nn.Linear(600, 128)
        self.hidden1 = nn.Linear(128, 64)
        self.hidden2 = nn.Linear(64,8)
        self.output = nn.Linear(8,1)
    
    def forward(self,x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.input_layer(x))
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = self.output(x)
        return x


# # DNN Optimised
# class DNN(nn.Module):
#     def __init__(self):
#         super(DNN, self).__init__()
#         self.input_layer = nn.Linear(600, 128)
#         self.hidden1 = nn.Linear(128, 64)
#         self.hidden2 = nn.Linear(64, 32)
#         self.hidden3 = nn.Linear(32, 16)
#         self.output = nn.Linear(16, 1)
#         self.dropout = nn.Dropout(0.3)
#         self.bn1 = nn.BatchNorm1d(128)
#         self.bn2 = nn.BatchNorm1d(64)

#     def forward(self,x):
#         x = x.view(x.size(0), -1)  # Flatten input
#         x = F.relu(self.bn1(self.input_layer(x)))
#         x = F.relu(self.bn2(self.hidden1(x)))
#         x = self.dropout(F.relu(self.hidden2(x)))
#         x = F.relu(self.hidden3(x))
#         x = self.output(x)
#         return x

def InitWeights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)