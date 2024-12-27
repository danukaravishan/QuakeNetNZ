import torch
import torch.nn as nn
import torch.nn.functional as F

class conv1x1(nn.Module):
    """
        Two 1x1 convolutional layer with relu Activation
    """
    def __init__(self, in_channels, out_channels, kernel_size=7, stride=1, padding=3):
        super(conv1x1, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x

class DownSampling(nn.Module):
    """
        Down Sampling with Convolution and Max Pooling
        Convolution + Stride + Relu Activation
    """
    def __init__(self, in_channels, out_channels, kernel_size=7, stride=4, padding_down=0, padding_conv=3):
        super(DownSampling, self).__init__()
        self.downsample = nn.Conv1d(in_channels, in_channels ,kernel_size, stride, padding=padding_down)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride =1, padding=padding_conv)

    def forward(self, x):
        x = F.relu(self.downsample(x))
        x = F.relu(self.conv(x))
        return x
    
class UpSampling(nn.Module):
    """
        Up Sampling with Convolution and Upsample
        Convolution + Stride + Relu Activation
    """
    def __init__(self, in_channels, out_channels, kernel_size=7, stride=4, padding_down=0, padding_conv = 3):
        super(UpSampling, self).__init__()
        self.upsample = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding=padding_down)
        self.conv = nn.Conv1d(out_channels*2, out_channels, kernel_size, stride =1, padding=padding_conv)

    def forward(self, x, skip_x):
        x = F.relu(self.upsample(x))
        x = torch.cat(x, skip_x, dims= 1)
        x = F.relu(self.conv(x))
        return x

class uNet(nn.Module):
    """
        Phasenet Architecture for QuakeNetNZ 
        1D time series data in 4s Window with 50Hz Sampling Rate
        Input: 3x200(3 channels(E=East,N=North,Z=Vertical), 50x4 samples)
        Output: 3x200(Probabilities for P-pick, S-pick and noise)
    """
    def __init__(self, in_channels=3, out_channels=3):
        super(uNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.down_channels = [8,11,16,22,32]
        self.up_channels = self.down_channels[::-1]

        self.down_layers = nn.ModuleList()
        in_channels = self.in_channels

        # Down Sampling Layers
        for i in range(len(self.down_channels)):
            ins = self.in_channels if i == 0 else outs
            outs = base_filters
            self.down_layers.append(DownSampling(in_channels=ins, out_channels=outs))
            in_channels = out_channels
            base_filters *= 2
        
        self.up_layers = nn.ModuleList()

        # Up Sampling Layers
        for i in range(depth-2):
            ins = outs
            outs = ins//2
            self.up_layers.append(UpSampling(in_channels=ins, out_channels=outs))
    
    def forward(self,x):
        skip_connections = []
        for layer in self.down_layers:
            x = layer(x)
            skip_connections.append(x)
        
        for i, layer in enumerate(self.up_layers):
            x = layer(x, skip_connections[-i-1])

        return x

        


