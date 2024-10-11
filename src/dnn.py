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
        return x


