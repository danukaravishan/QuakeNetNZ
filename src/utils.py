# Os utils
import os

# Data handling
import random
import pandas as pd
import h5py
import numpy as np
import obspy
from obspy import Trace
from obspy import Stream
from scipy.signal import decimate
from obspy import UTCDateTime
from datetime import datetime
import uuid

## Plotting
import matplotlib.pyplot as plt

## ML libraries
from cnn import PWaveCNN, MobileNet1D
from dnn import DNN
from dnn import InitWeights
from cred import CRED
from unet import uNet
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn

from report   import *

# Signal processing
import scipy 
import scipy.signal as signal


# Seisbench 
import seisbench.models as sbm


def find_latest_file(directory, prefix, extension):
    files = [f for f in os.listdir(directory) if f.startswith(prefix) and f.endswith(extension)]
    if not files:
        raise ValueError(f"No files with prefix '{prefix}' and extension '{extension}' found in {directory}")
    latest_file = max(files, key=lambda x: os.path.getctime(os.path.join(directory, x)))
    return latest_file


def getLatestModelName(cfg):
    # Fine the latest file name
    directory = cfg.MODEL_PATH  # Change to your model directory
    model_prefix = (cfg.MODEL_TYPE.name).lower()
    model_extension = ".pt"
    latest_model_file = find_latest_file(directory, model_prefix, model_extension)
    return latest_model_file