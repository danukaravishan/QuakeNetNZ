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
import seaborn as sns
from statsmodels.nonparametric.smoothers_lowess import lowess

## ML libraries
from cnn import PWaveCNN
from models import PDetector, MobileNet1D, TFEQ, CNNRNN
from dnn import DNN
from dnn import InitWeights
from cred import CRED
from unet import uNet
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
from dataprep import pre_proc_data

from report   import *

# Signal processing
import scipy 
import scipy.signal as signal
from sklearn.preprocessing import MinMaxScaler

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


def plot_waveform_with_picks(event_id, hdf5_file):
    """
    Plots the waveform data and marks the P and S wave picks using their attributes.

    Args:
        dataset (h5py.Dataset): The waveform dataset containing the data and attributes.
        sampling_rate (int): The sampling rate of the waveform data.
    """

    with h5py.File(hdf5_file, 'r') as hdf:
        dataset = hdf["data"].get(event_id)

        data = np.array(dataset)
        sampling_rate = dataset.attrs.get("sampling_rate", None)
        data_pre_proc = pre_proc_data(data, sampling_rate=sampling_rate)

        # Extract channels from attributes and split them
        channels_attr = dataset.attrs.get("channels", "")
        channels = channels_attr.split(",") if channels_attr else [f"Channel {i+1}" for i in range(len(data))]

        # Get P and S wave pick times from attributes
        p_pick_time = dataset.attrs.get("p_arrival_sample", None)
        # Convert pick times to indices if they exist
        p_arrival_index = int(p_pick_time) if p_pick_time is not None and not np.isnan(p_pick_time) else None

        epicentral_distance = dataset.attrs.get("epicentral_distance", None)
        magnitude = dataset.attrs.get("magnitude", None)
        
        plt.figure(figsize=(12, 6))

        # Plot each channel in the dataset
        plt.subplot(2, 1, 1)
        for i, channel in enumerate(data):
            plt.plot(channel, label=channels[i] if i < len(channels) else f"Channel {i+1}")

        # Mark the P-wave arrival if available
        if p_arrival_index is not None:
            plt.axvline(x=p_arrival_index, color='r', linestyle='--', label='P-wave Pick')
            if sampling_rate is not None:
                plt.axvline(x=p_arrival_index - sampling_rate * 1, color='r', linestyle=':', label='P-wave Window Start')
                plt.axvline(x=p_arrival_index + sampling_rate * 1, color='r', linestyle=':', label='P-wave Window End')

        plt.title(f"Waveform with P Wave Pick, Event {event_id}, Magnitude {magnitude}, Epicentral Distance {epicentral_distance}Km")
        plt.xlabel("Sample Index")
        plt.ylabel("Acceleration (m/s²)")
        plt.legend()
        plt.grid()

        plt.subplot(2, 1, 2)
        for i, channel in enumerate(data_pre_proc):
            # Trim the first and last 20 samples
            trimmed_channel = channel[30:-20]
            plt.plot(trimmed_channel, label=channels[i] if i < len(channels) else f"Channel {i+1}")
        if p_arrival_index is not None:
            plt.axvline(x=p_arrival_index - 20, color='r', linestyle='--', label='P-wave Pick')
            if sampling_rate is not None:
                plt.axvline(x=p_arrival_index - sampling_rate * 1 - 20, color='r', linestyle=':', label='P-wave Window Start')
                plt.axvline(x=p_arrival_index + sampling_rate * 1 - 20, color='r', linestyle=':', label='P-wave Window End')

        plt.title(f"Pre-processed Waveform with P Wave Pick, Event {event_id}")
        plt.xlabel("Sample Index")
        plt.ylabel("Acceleration (m/s²)")
        plt.legend()
        plt.grid()

        plt.tight_layout()
        plt.show()




def plot_2s_waveform(event_id, hdf5_file):
    """
    Plots the 3 components of the waveform data for the given event ID.

    Args:
        event_id (str): The ID of the event to plot.
        hdf5_file (str): Path to the HDF5 file containing the waveform data.
    """
    with h5py.File(hdf5_file, 'r') as hdf:
        dataset = hdf["positive_samples_p"].get(event_id)
        data = np.array(dataset)

        plt.figure(figsize=(4, 6))  # Adjusted figure size for better visualization
        for i in range(3):  # Assuming 3 components
            plt.subplot(3, 1, i + 1)
            plt.plot(data[i], label=f"Component {i + 1}")
            plt.title(f"Component {i + 1} - Event {event_id}")
            plt.xlabel("Sample Index")
            plt.ylabel("Amplitude")
            plt.legend()
            plt.grid()

        plt.tight_layout()
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


def plot_model_variations(csv_file, save_path=None):
    """
    Reads the model details CSV and plots:
    1. Parameter Count vs Precision
    2. Parameter Count vs Recall
    3. Parameter Count vs F1 Score
    4. FLOPs vs Recall
    If save_path is provided, saves the figure to that path instead of showing it.
    """
    df = pd.read_csv(csv_file)
    # Use correct column names from the CSV
    param_col = 'Parameter Count' if 'Parameter Count' in df.columns else 'Model Parameters'
    flops_col = 'flops' if 'flops' in df.columns else 'FLOPs'
    df[param_col] = df[param_col].astype(str).str.replace(',', '').astype(float)
    df['Precision'] = df['Precision'].astype(str).str.replace('%', '').astype(float)
    df['Recall'] = df['Recall'].astype(str).str.replace('%', '').astype(float)
    df['F1 Score'] = df['F1 Score'].astype(str).str.replace('%', '').astype(float)
    df[flops_col] = df[flops_col].astype(str).str.replace(',', '').astype(float)

    plt.figure(figsize=(14, 10))
    plt.subplot(2,2,1)
    plt.scatter(df[param_col], df['Precision'], c='b', label='Precision')
    plt.xlabel('Parameter Count')
    plt.ylabel('Precision (%)')
    plt.title('Parameter Count vs Precision')
    plt.grid(True)

    plt.subplot(2,2,2)
    plt.scatter(df[param_col], df['Recall'], c='g', label='Recall')
    plt.xlabel('Parameter Count')
    plt.ylabel('Recall (%)')
    plt.title('Parameter Count vs Recall')
    plt.grid(True)

    plt.subplot(2,2,3)
    plt.scatter(df[param_col], df['F1 Score'], c='r', label='F1 Score')
    plt.xlabel('Parameter Count')
    plt.ylabel('F1 Score (%)')
    plt.title('Parameter Count vs F1 Score')
    plt.grid(True)

    plt.subplot(2,2,4)
    plt.scatter(df[flops_col], df['Recall'], c='m', label='Recall')
    plt.xlabel('FLOPs')
    plt.ylabel('Recall (%)')
    plt.title('FLOPs vs Recall')
    plt.grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()




def plot_scaled_param_flops_vs_metrics(csv_file, save_path=None):
    """
    Scales Parameter Count and FLOPs to [0,1], forms a single composite value (sum or weighted sum),
    and plots this value vs Precision, Recall, and F1 Score.
    """
    df = pd.read_csv(csv_file)
    param_col = 'Parameter Count' if 'Parameter Count' in df.columns else 'Model Parameters'
    flops_col = 'flops' if 'flops' in df.columns else 'FLOPs'
    df[param_col] = df[param_col].astype(str).str.replace(',', '').astype(float)
    df[flops_col] = df[flops_col].astype(str).str.replace(',', '').astype(float)
    df['Precision'] = df['Precision'].astype(str).str.replace('%', '').astype(float)
    df['Recall'] = df['Recall'].astype(str).str.replace('%', '').astype(float)
    df['F1 Score'] = df['F1 Score'].astype(str).str.replace('%', '').astype(float)

    # Scale both columns to [0, 1]
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(df[[param_col, flops_col]].values)
    # Composite value: sum of scaled param count and flops
    df['ScaledParamFLOPs'] = X_scaled[:, 0] + X_scaled[:, 1]

    plt.figure(figsize=(10, 6))
    # plt.subplot(1,3,1)
    # plt.scatter(df['ScaledParamFLOPs'], df['Precision'], c='b', alpha=0.7)
    # plt.xlabel('Scaled Param+FLOPs')
    # plt.ylabel('Precision (%)')
    # plt.title('Scaled Param+FLOPs vs Precision')
    # plt.grid(True)

    # plt.subplot(1,3,2)
    # plt.scatter(df['ScaledParamFLOPs'], df['Recall'], c='g', alpha=0.7)
    # plt.xlabel('Scaled Param+FLOPs')
    # plt.ylabel('Recall (%)')
    # plt.title('Scaled Param+FLOPs vs Recall')
    # plt.grid(True)

    #plt.subplot(1,3,3)
    plt.scatter(df['ScaledParamFLOPs'], df['F1 Score'], c='r', alpha=0.7)
    plt.xlabel('Complexity Index')
    plt.ylabel('F1 Score (%)')
    plt.title('Complexity Index vs F1 Score')
    plt.grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()



def plot_complexity_index_vs_f1(csv_file, save_path=None, bins=40):
    """
    Publication-quality plot: Complexity Index vs F1 Score, with a max-value envelope curve.
    The max curve is computed by binning the Complexity Index and taking the max F1 Score in each bin.
    The x-axis grid values are annotated with the actual parameter count and FLOPs for the bin centers.
    """
    df = pd.read_csv(csv_file)
    param_col = 'Parameter Count' if 'Parameter Count' in df.columns else 'Model Parameters'
    flops_col = 'flops' if 'flops' in df.columns else 'FLOPs'
    df[param_col] = df[param_col].astype(str).str.replace(',', '').astype(float)
    df[flops_col] = df[flops_col].astype(str).str.replace(',', '').astype(float)
    df['F1 Score'] = df['F1 Score'].astype(str).str.replace('%', '').astype(float)

    # Scale
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(df[[param_col, flops_col]].values)
    df['Complexity Index'] = X_scaled[:, 0] + X_scaled[:, 1]

    # Set style
    sns.set(style="whitegrid", font_scale=1.5)
    plt.figure(figsize=(10, 6), dpi=300)

    # Scatter
    plt.scatter(df['Complexity Index'], df['F1 Score'], c='crimson', alpha=0.7, edgecolor='k', s=60, label='Model')

    # Max curve
    bins_edges = np.linspace(df['Complexity Index'].min(), df['Complexity Index'].max(), bins+1)
    bin_centers = 0.5 * (bins_edges[:-1] + bins_edges[1:])
    max_f1 = [df[(df['Complexity Index'] >= bins_edges[i]) & (df['Complexity Index'] < bins_edges[i+1])]['F1 Score'].max() for i in range(bins)]

    # For each bin center, get the median (or mean) param count and flops in that bin
    param_ticks = []
    flops_ticks = []
    for i in range(bins):
        bin_df = df[(df['Complexity Index'] >= bins_edges[i]) & (df['Complexity Index'] < bins_edges[i+1])]
        if not bin_df.empty:
            param_ticks.append(int(bin_df[param_col].median()))
            flops_ticks.append(int(bin_df[flops_col].median()))
        else:
            param_ticks.append(None)
            flops_ticks.append(None)

    plt.plot(bin_centers, max_f1, color='navy', linewidth=2.5, label='Max Envelope')

    # Labels and title
    plt.xlabel('Complexity Index', fontsize=18, fontweight='bold')
    plt.ylabel('F1 Score (%)', fontsize=18, fontweight='bold')
    plt.title('Complexity Index vs F1 Score', fontsize=20, fontweight='bold')
    plt.legend(fontsize=14)
    plt.tight_layout()

    # Set custom x-ticks with param count and flops
    xticks = plt.xticks()[0]
    xtick_labels = []
    for x in xticks:
        # Find the closest bin center
        idx = (np.abs(bin_centers - x)).argmin()
        if param_ticks[idx] is not None and flops_ticks[idx] is not None:
            label = f"{x:.2f}\nP:{param_ticks[idx]}\nF:{flops_ticks[idx]//1000}K"
        else:
            label = f"{x:.2f}"
        xtick_labels.append(label)
    plt.xticks(xticks, xtick_labels, rotation=0, fontsize=12)

    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def get_event_details(hdf5_path, event_id):
    """
    Print station, magnitude, and epicentral distance for all samples related to a specific earthquake event.
    
    Parameters:
    - hdf5_path: Path to the HDF5 file
    - event_id: Earthquake event ID (e.g., "20161114225955495")
    """
    with h5py.File(hdf5_path, 'r') as f:
        group = f['positive_samples_p']
        found = False

        print(f"Details for event ID: {event_id}")
        print("-" * 50)
        for key in group.keys():
            if key.startswith(event_id):
                ds = group[key]
                station = ds.attrs.get('station', b'').decode() if isinstance(ds.attrs.get('station', None), bytes) else ds.attrs.get('station', '')
                magnitude = ds.attrs.get('magnitude', 'N/A')
                epicentral_distance = ds.attrs.get('epicentral_distance', 'N/A')

                print(f"Station: {station}, Magnitude: {magnitude}, Distance: {epicentral_distance} km")
                found = True
        
        if not found:
            print("No samples found for this event.")


def plot_magnitude_distribution(train_hdf5, test_hdf5, save_path=None, test_val_split_ratio=0.5, group_name='positive_samples_p', normalize=True):
    """
    Plots the magnitude distribution for training, validation, and test sets as grouped bars (side-by-side),
    using the same splitting logic as in the main code.
    - train_hdf5: path to training HDF5 file
    - test_hdf5: path to test HDF5 file
    - test_val_split_ratio: fraction of test set to use for val (rest is test)
    - group_name: group in HDF5 file containing events (default 'positive_samples_p')
    - normalize: if True, plot fraction in each bin; else, plot counts
    """
    import h5py
    import numpy as np
    import matplotlib.pyplot as plt

    def get_magnitudes(hdf5_path, group_name):
        mags = []
        with h5py.File(hdf5_path, 'r') as hdf:
            group = hdf[group_name]
            for event_id in group:
                ds = group[event_id]
                mag = ds.attrs.get('magnitude', None)
                if mag is not None:
                    mags.append(mag)
        return np.array(mags)

    # Get train and test magnitudes
    train_mags = get_magnitudes(train_hdf5, group_name)
    test_mags = get_magnitudes(test_hdf5, group_name)

    # Split test set into val and test using the same method as the main code
    np.random.seed(42)
    n_val = int(test_val_split_ratio * len(test_mags))
    idx = np.random.permutation(len(test_mags))
    val_idx = idx[:n_val]
    test_idx = idx[n_val:]
    val_mags = test_mags[val_idx]
    test_mags = test_mags[test_idx]

    # Define bins (integer magnitudes)
    all_mags = np.concatenate([train_mags, val_mags, test_mags])
    min_mag = int(np.floor(all_mags.min()))
    max_mag = int(np.ceil(all_mags.max()))
    bins = np.arange(min_mag, max_mag + 2) - 0.5  # Centered bins
    bin_centers = np.arange(min_mag, max_mag + 1)

    # Histogram counts
    train_hist, _ = np.histogram(train_mags, bins=bins)
    val_hist, _ = np.histogram(val_mags, bins=bins)
    test_hist, _ = np.histogram(test_mags, bins=bins)

    if normalize:
        train_hist = train_hist / train_hist.sum() if train_hist.sum() > 0 else train_hist
        val_hist = val_hist / val_hist.sum() if val_hist.sum() > 0 else val_hist
        test_hist = test_hist / test_hist.sum() if test_hist.sum() > 0 else test_hist
        ylabel = 'Fraction in Bin'
    else:
        ylabel = 'Frequency'

    width = 0.25
    x = bin_centers
    plt.figure(figsize=(10,5))
    plt.bar(x - width, train_hist, width=width, color='slateblue', alpha=0.7, label='Training Data', align='center')
    plt.bar(x, val_hist, width=width, color='goldenrod', alpha=0.7, label='Validation Data', align='center')
    plt.bar(x + width, test_hist, width=width, color='tomato', alpha=0.7, label='Test Data', align='center')

    plt.xlabel('Magnitude', fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.xticks(x)
    plt.legend(fontsize=12)
    plt.title('Magnitude Distribution of Training, Validation, and Test Sets', fontsize=16, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
