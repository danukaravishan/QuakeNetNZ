import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
import random
import numpy as np
import secrets
import pywt
import math

# # Random seed

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# If using CUDA (GPU)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

###################################################################################################

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


###################################################################################################
# PWave Detector using 4 second - Irshard et al. 2022
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
    
###################################################################################################

# TFEQ Implementation
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].to(x.device)
        return x

# Feed Forward Network (MLP)
class FeedForward(nn.Module):
    def __init__(self, d_model, hidden_dim):
        super().__init__()
        self.linear1 = nn.Linear(d_model, hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, d_model)

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))

# Transformer Encoder Block
class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, ff_hidden_dim):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, ff_hidden_dim)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_output, _ = self.self_attn(x, x, x)
        x = self.norm1(x + attn_output)
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        return x


class TFEQMy(nn.Module):
    def __init__(self, input_dim=3, seq_len=200, embed_dim=64, num_heads=8, ff_hidden_dim=128, encoder_layers=1, mlp_output_dim=200):
        super().__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=seq_len)

        self.encoder_layers = nn.Sequential(*[
            EncoderBlock(embed_dim, num_heads, ff_hidden_dim)
            for _ in range(encoder_layers)
        ])

        self.decoder_fc1 = nn.Linear(embed_dim, embed_dim)
        self.decoder_fc2 = nn.Linear(embed_dim, input_dim)  # back to (200, 3)
        self.mlp = nn.Sequential(
           nn.Linear(seq_len * input_dim, mlp_output_dim),
           nn.ReLU(),
           nn.Linear(mlp_output_dim, 1),  # One output!
           nn.Sigmoid()  # probability output
        )

        random_tag = str(secrets.randbelow(9000) + 1000)
        self.model_id = "tfeq_" + datetime.now().strftime("%Y%m%d_%H%M") + "_" + random_tag

    def forward(self, x):
        #if x.shape[1] == 3 and x.shape[2] != 3:
        x = x.permute(0, 2, 1) # Since the base code is sending data as (batch, 3, 200)
        # x: (batch, seq_len, 3)
        x = self.embedding(x)                       # (batch, 200, 64)
        x = self.pos_encoder(x)                     # Add positional encoding
        x = self.encoder_layers(x)                  # Transformer encoder blocks
        x = F.relu(self.decoder_fc1(x))             # Transformer decoder (2 FC layers)
        x = self.decoder_fc2(x)                     # (batch, 200, 3)
        x = x.view(x.size(0), -1)                   # flatten to (batch, 600)
        out = self.mlp(x)                           # MLP with 200 neurons -> 1-class output
        return out           # final output probabilities



###################### TFEQ copied model #################
class TFEQ(nn.Module):
    def __init__(self, channel=3, dmodel=64, nhead=8, dim_feedforward=32, num_layers=1, time_in=200):
        super(TFEQ, self).__init__()
        self.input_embedding = nn.Linear(channel, dmodel)
        self.temporal_encoder_layer = nn.TransformerEncoderLayer(
            dmodel, nhead, dim_feedforward)
        self.temporal_encoder = nn.TransformerEncoder(
            self.temporal_encoder_layer, num_layers)
        self.decoder = nn.Linear(dmodel, 1)
        self.temporal_pos = torch.arange(0, time_in).cuda()
        self.temporal_pe = nn.Embedding(time_in, dmodel)
        self.fla = nn.Flatten()
        self.dp = nn.Dropout(p=0.5)
        self.fc = nn.Linear(time_in, 2)
        random_tag = str(secrets.randbelow(9000) + 1000)
        self.model_id = "tfeq_" + datetime.now().strftime("%Y%m%d_%H%M") + "_" + random_tag

    def forward(self, x):
        x = x.permute(0, 2, 1) # Input datashape of needs to be transposed to (batch, time, channel)
        b, t,  c = x.shape  # [B, 200, 3]
        x = self.input_embedding(x)  # [B, 200,  64]
        t_pe = self.temporal_pe(self.temporal_pos).expand(b, t, -1)
        x = x + t_pe
        x = self.temporal_encoder(x)
        x = self.decoder(x)  # [B, 1, 200, 1]
#         print(x.shape)
        x = self.fla(x)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)
    


############################################################################################
### CNNRNN architecture
class CNNBranch(nn.Module):
    def __init__(self):
        super(CNNBranch, self).__init__()
        self.conv1 = nn.Conv1d(3, 16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.pool = nn.AdaptiveAvgPool1d(1)  # Output: (batch, 32, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))      # (batch, 16, 100)
        x = F.relu(self.conv2(x))      # (batch, 32, 100)
        x = self.pool(x)               # (batch, 32, 1)
        return x.squeeze(-1)           # (batch, 32)

class RNNBranch(nn.Module):
    def __init__(self, hidden_size=32):
        super(RNNBranch, self).__init__()
        self.lstm = nn.LSTM(input_size=3, hidden_size=hidden_size, batch_first=True)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch, 100, 3)
        _, (h_n, _) = self.lstm(x)
        return h_n.squeeze(0)  # (batch, 32)


class CNNRNN(nn.Module):
    def __init__(self):
        super(CNNRNN, self).__init__()
        self.cnn_branch = CNNBranch()
        self.rnn_branch = RNNBranch()
        self.alpha = nn.Parameter(torch.tensor(0.5))  # learnable fusion weight

        self.fc1 = nn.Linear(32, 16)
        self.fc2 = nn.Linear(16, 1)
        random_tag = str(secrets.randbelow(9000) + 1000)
        self.model_id = "cnnrnn_" + datetime.now().strftime("%Y%m%d_%H%M") + "_" + random_tag

    def forward(self, x):
        cnn_out = self.cnn_branch(x)  # (batch, 32)
        rnn_out = self.rnn_branch(x)  # (batch, 32)

        fused = self.alpha * cnn_out + (1 - self.alpha) * rnn_out
        x = F.relu(self.fc1(fused))
        return torch.sigmoid(self.fc2(x))  # (batch, 1)



# if __name__ == "__main__":
#     model = TFEQ()
#     dummy_input = torch.randn(64, 3, 200)  # batch of 8 samples
#     output = model(dummy_input)
#     print("Output shape:", output.shape) 