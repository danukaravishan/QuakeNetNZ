import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime

class conv1x1(nn.Module):
    """
        Two 1x1 convolutional layer with relu Activation
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(conv1x1, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding)
        
    def forward(self, x):
        # print(f"Input size - x shape: {x.shape}")
        x = F.relu(self.conv1(x))
        # print(f"After first convolution - x shape: {x.shape}")
        x = F.relu(self.conv2(x))
        # print(f"After second convolution - x shape: {x.shape}")
        return x
    
class lastconv1x1(nn.Module):
    """
        1x1 convolutional layer with and softmax Activation
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(lastconv1x1, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        
    def forward(self, x):
        # print("Last Convolution")
        x = F.sigmoid(self.conv(x))
        # print(f"After last convolution - x shape: {x.shape}")
        return x
    
class DownSampling(nn.Module):
    """
        Down Sampling with Convolution and Max Pooling
        Convolution + Stride + Relu Activation
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding_down=1):
        super(DownSampling, self).__init__()
        self.downsample = nn.Conv1d(in_channels, in_channels ,kernel_size, stride, padding=padding_down)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride =1, padding=1)

    def forward(self, x):
        # print(f"Input size before downsampling- x shape: {x.shape}")
        x = F.relu(self.downsample(x))
        # print(f"After down sampling - x shape: {x.shape}")
        x = F.relu(self.conv(x))
        # print(f"After down sampling + convolution - x shape: {x.shape}")
        return x
    
class UpSampling(nn.Module):
    """
        Up Sampling with Convolution and Upsample
        Convolution + Stride + Relu Activation
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding_up=1):
        super(UpSampling, self).__init__()
        self.upsample = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding=padding_up)
        self.conv = nn.Conv1d(out_channels*2, out_channels, kernel_size, stride =1, padding=1)

    def pad_tensor(self, x, skip_x):
        target_size = skip_x.size()[2]
        input_size = x.size()[2]
        delta = target_size - input_size
        # print(f"Delta: {delta}")
        padding_left = delta % 2
        # print(f"Padding: {padding_left}")
        padding_right = delta-padding_left
        x = F.pad(x, (padding_left,padding_right))
        return x
    
    def forward(self, x, skip_x):
        # print(f"Before upsampling - x shape: {x.shape}, skip_x shape: {skip_x.shape}")
        x = F.relu(self.upsample(x))
        if(x.size()[2] != skip_x.size()[2]):
            # print("I am here")
            x = self.pad_tensor(x, skip_x)
        # print(f"After upsampling - x shape: {x.shape}")
        x = torch.cat((x, skip_x), dim= 1)
        # print(f"After concatenation - x shape: {x.shape}")
        x = F.relu(self.conv(x))
        # print(f"After convolution - x shape: {x.shape}")
        return x
    

class uNet(nn.Module):
    """
        Phasenet Architecture for QuakeNetNZ 
        1D time series data in 4s Window with 50Hz Sampling Rate
        Input: 3x200(3 channels(E=East,N=North,Z=Vertical), 50x4 samples)
        Output: 3x200(Probabilities for P-pick, S-pick and noise)
    """
    def __init__(self, in_channels=3, out_channels=1, model_id=""):
        super(uNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.down_channels = [8,11,16,22,32]
        self.up_channels = self.down_channels[::-1]

        self.down_layers = nn.ModuleList()

        # Model ID with timestamp if not provided
        self.model_id = "unet_" + datetime.now().strftime("%Y%m%d_%H%M") if model_id == "" else model_id

        # Down Sampling Layers
        for i in range(len(self.down_channels)):
            ins = self.in_channels if i == 0 else self.down_channels[i-1]
            outs = self.down_channels[i]
            if i==0:
                self.down_layers.append(conv1x1(in_channels=ins, out_channels=outs))
                # print(f'in channels:{ins}, out channels: {outs}')
            else:
                self.down_layers.append(DownSampling(in_channels=ins, out_channels=outs))
                # print(f'in channels:{ins}, out channels: {outs}')

        self.up_layers = nn.ModuleList()

        # Up Sampling Layers
        for i in range(len(self.down_channels)-1):
            ins = self.up_channels[i]
            outs = self.up_channels[i+1]
            self.up_layers.append(UpSampling(in_channels=ins, out_channels=outs))
        self.up_layers.append(lastconv1x1(in_channels=self.up_channels[-1], out_channels=self.out_channels))
    
    def forward(self,x):
        skip_connections = []
        # print(f"Input size: {x.size()}")
        for layer in self.down_layers:
            x = layer(x)
            skip_connections.append(x)
        
        for i, layer in enumerate(self.up_layers[:-1]):
            skip_x = skip_connections[-(i+2)]
            x = layer(x, skip_x)
            # print(f"x shape: {x.shape}, skip_x shape: {skip_x.shape}")
        
        # Final Output Layer
        x = self.up_layers[-1](x)

        return x
