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
from cnn import PWaveCNN, MobileNet1D, PDetector
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


def plot_waveform_with_picks(dataset):
    """
    Plots the waveform data and marks the P and S wave picks using their attributes.

    Args:
        dataset (h5py.Dataset): The waveform dataset containing the data and attributes.
        sampling_rate (int): The sampling rate of the waveform data.
    """
    # Extract waveform data
    data = np.array(dataset)

    # Get P and S wave pick times from attributes
    p_pick_time = dataset.attrs.get("trace_p_arrival_sample", None)
    s_pick_time = dataset.attrs.get("trace_s_arrival_sample", None)
    # Convert pick times to indices if they exist
    p_arrival_index = int(p_pick_time) if not np.isnan(p_pick_time) else None
    s_arrival_index = int(s_pick_time) if not np.isnan(s_pick_time) else None

    plt.figure(figsize=(12, 6))

    # Plot each channel in the dataset
    for i, channel in enumerate(data):
        plt.plot(channel, label=f"Channel {i+1}")

    # Mark the P-wave arrival if available
    if p_arrival_index is not None:
        plt.axvline(x=p_arrival_index, color='r', linestyle='--', label='P-wave Pick')

    # Mark the S-wave arrival if available
    if s_arrival_index is not None:
        plt.axvline(x=s_arrival_index, color='b', linestyle='--', label='S-wave Pick')

    plt.title("Waveform with P and S Wave Picks")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid()
    plt.show()


def count_records_in_hdf5_groups(hdf5_file_path):
    """
    Prints the number of records in each group of an HDF5 file.

    Args:
        hdf5_file_path (str): Path to the HDF5 file.
    """
    with h5py.File(hdf5_file_path, 'r') as hdf:
        for group_name in hdf.keys():
            group = hdf[group_name]
            record_count = len(group.keys())
            print(f"Group: {group_name}, Records: {record_count}")


def check_duplicate_event_ids(hdf5_file_path):
    """
    Checks for duplicate event IDs in each group of an HDF5 file and prints the results.

    Args:
        hdf5_file_path (str): Path to the HDF5 file.
    """
    with h5py.File(hdf5_file_path, 'r') as hdf:
        for group_name in hdf.keys():
            group = hdf[group_name]
            event_ids = list(group.keys())
            duplicates = set([event_id for event_id in event_ids if event_ids.count(event_id) > 1])

            if duplicates:
                print(f"Group: {group_name} has duplicate event IDs: {duplicates}")
            else:
                print(f"Group: {group_name} has no duplicate event IDs.")