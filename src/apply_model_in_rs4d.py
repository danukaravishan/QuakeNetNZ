
import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from utils import *
from config import Config
from extract_window_db import *
from plots import plot_stations_on_nz_map
import torch
from obspy import read
from scipy.signal import resample
from dataprep import pre_proc_data, normalize_data
import torch.nn.functional as F

def apply_model_to_mssed_files(cfg, mssed_dir, output_dir, model, nncfg, sampling_rate=50, window_size=100, stride=10):
    """
    Apply the trained model to all mssed files in a directory, plot original waveforms and model predictions.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for fname in os.listdir(mssed_dir):
        if not (fname.endswith('.mseed') or fname.endswith('.mssed') or fname.endswith('.miniseed') or fname.endswith('.ms')):
            continue
        fpath = os.path.join(mssed_dir, fname)
        try:
            st = read(fpath)
        except Exception as e:
            print(f"Error reading {fname}: {e}")
            continue
        if len(st) < 3:
            print(f"Skipping {fname}: less than 3 components")
            continue
        # Sort by channel name to get Z, N, E order if possible
        st.sort(['channel'])
        # Ensure all three traces have the same length
        min_len = min(len(tr.data) for tr in st[:3])
        data = np.array([tr.data[:min_len] for tr in st[:3]])
        original_rate = st[0].stats.sampling_rate
        n_samples = data.shape[1]
        # Resample if needed
        if original_rate != sampling_rate:
            num_samples = int(n_samples * sampling_rate / original_rate)
            data = np.array([resample(ch, num_samples) for ch in data])
        data_pp = pre_proc_data(data, original_rate)
        if original_rate != sampling_rate:
            data_pp = np.array([resample(ch, data.shape[1]) for ch in data_pp])
        total_samples = data.shape[1]
        total_seconds = total_samples / sampling_rate
        timestamps = np.arange(0, total_seconds, 1 / sampling_rate)
        # Model inference
        real_time_window = int(2 * sampling_rate)
        time_preds = []
        raw_output = []
        classified_output = []
        for current_time in range(0, total_samples, stride):
            start = max(0, current_time - real_time_window)
            segment = data[:, start:current_time]
            if segment.shape[1] < window_size:
                continue
            segment = normalize_data(segment)
            input_tensor = torch.tensor(segment, dtype=torch.float32).unsqueeze(0)
            output = model(input_tensor.to(device))
            if cfg.MODEL_TYPE == cfg.MODEL_TYPE.TFEQ:
                raw_output.append((F.softmax(output, dim=1)[:,1]).item())
                cl_out = output.argmax(dim=1)
            else:
                cl_out = output > 0.9
                raw_output.append(output.item())
            classified_output.append(cl_out.item())
            time_preds.append(current_time / sampling_rate)
        classified_output = np.array(classified_output)
        # Plot
        fig, axes = plt.subplots(4, 1, figsize=(16, 16), sharex=True)
        colors = ["b", "g", "r"]
        labels = ["Z Component", "N Component", "E Component"]
        for i in range(3):
            axes[0].plot(timestamps, data[i], color=colors[i], label=labels[i])
        axes[0].set_title(f"Waveform - {fname}")
        axes[0].set_ylabel("Amplitude")
        axes[0].legend()
        for i in range(3):
            axes[1].plot(timestamps, data_pp[i], color=colors[i], label=labels[i])
        axes[1].set_title("Pre-processed Waveform")
        axes[1].set_ylabel("Amplitude")
        axes[1].legend()
        axes[2].vlines(time_preds, ymin=0, ymax=raw_output, color="k", linewidth=1)
        axes[2].plot(time_preds, raw_output, 'o', color="r", markersize=3, label="Model Predictions - Earthquake")
        axes[2].axhline(y=0.9, color='blue', linestyle='--', label="Cutoff Threshold (0.9)")
        axes[2].set_title("Raw Model Output")
        axes[2].set_xlabel("Time (seconds)")
        axes[2].legend()
        axes[3].vlines(time_preds, ymin=0, ymax=classified_output, color="k", linewidth=1)
        axes[3].plot(time_preds, classified_output, 'o', color="k", markersize=3, label="Model Predictions - Noise")
        axes[3].set_title("Classified Output")
        axes[3].set_xlabel("Time (seconds)")
        axes[3].set_ylabel("Prediction (scaled)")
        axes[3].legend()
        plt.tight_layout()
        output_file = os.path.join(output_dir, f"{os.path.splitext(fname)[0]}_model.png")
        plt.savefig(output_file)
        plt.close()
        print(f"Saved output for {fname} to {output_file}")



cfg = Config()
nncfg = NNCFG()
model = torch.load('models/ref/cnn_20250613_1327_1922.pt', map_location=torch.device('cpu'))  # Load your trained model on CPU
apply_model_to_mssed_files(
    cfg,
    mssed_dir='/Users/user/Downloads/mseed/',  # directory with your mssed files
    output_dir='plots/mssd_output.png',  # where to save the plots
    model=model,
    nncfg=nncfg,
    sampling_rate=50,  # or your model's expected rate
    window_size=100,   # or your model's expected window size
    stride=10          # or your preferred stride
)

#plot_miniseed_directory("/Users/user/Downloads/mseed/")