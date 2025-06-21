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
from obspy.clients.fdsn import Client
from obspy import UTCDateTime
import os
from obspy import read, UTCDateTime
from obspy.clients.fdsn import Client
from scipy.signal import resample
from dataprep import normalize_data
import numpy as np


def apply_model_to_mssd(mssd_file, model_file="models/ref/cnn_20250613_1327_1922.pt", sampling_rate=50, window_size=100, stride=10):

    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_file, map_location=torch.device(device))
    fname = os.path.basename(mssd_file)
    output_dir = "plots/rs4d_output/"
    try:
        st = read(mssd_file)
    except Exception as e:
        print(f"Error reading {fname}: {e}")
        return
    if len(st) < 3:
        print(f"Skipping {fname}: less than 3 components")
        return
    st.sort(['channel'])
    min_len = min(len(tr.data) for tr in st[:3])
    st = st[:3]
    st = st.copy()
    for i in range(3):
        st[i].data = st[i].data[:min_len]
    data = np.array([tr.data for tr in st[:3]])
    original_rate = st[0].stats.sampling_rate
    n_samples = data.shape[1]
    if original_rate != sampling_rate:
        num_samples = int(n_samples * sampling_rate / original_rate)
        data = np.array([resample(ch, num_samples) for ch in data])
    total_samples = data.shape[1]
    total_seconds = total_samples / sampling_rate
    timestamps = np.arange(0, total_seconds, 1 / sampling_rate)
    real_time_window = int(2 * sampling_rate)
    time_preds = []
    raw_output = []
    classified_output = []
    segment_count = 0
    for current_time in range(0, total_samples, stride):
        start = max(0, current_time - real_time_window)
        segment = data[:, start:current_time]
        if segment.shape[1] < window_size:
            continue
        segment_count += 1
        if segment_count <= 5:
            continue
        segment = normalize_data(segment)
        input_tensor = torch.tensor(segment, dtype=torch.float32).unsqueeze(0)
        output = model(input_tensor.to(device))

        if cfg.MODEL_TYPE == cfg.MODEL_TYPE.TFEQ:
            raw_output.append((F.softmax(output, dim=1)[:,1]).item())
            cl_out = output.argmax(dim=1)
        else:
            cl_out = output > 0.57
            raw_output.append(output.item())

        classified_output.append(cl_out.item())
        time_preds.append(current_time / sampling_rate)
    classified_output = np.array(classified_output)
    fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True)
    colors = ["b", "g", "r"]
    labels = ["Z Component", "N Component", "E Component"]
    for i in range(3):
        axes[i].plot(timestamps, data[i], color=colors[i], label=labels[i])
        axes[i].set_ylabel("Amplitude")
        axes[i].legend()
    station_name = st[0].stats.station
    axes[0].set_title(f"Waveform - M3.6, 2025p419621.{station_name}")
    axes[3].vlines(time_preds, ymin=0, ymax=raw_output, color="k", linewidth=1)
    axes[3].plot(time_preds, raw_output, 'o', color="r", markersize=3, label="Model Predictions - Earthquake")
    axes[3].axhline(y=0.57, color='blue', linestyle='--', label="Cutoff Threshold (0.57)")
    axes[3].plot(time_preds, classified_output, 'o', color="k", markersize=3, label="Model Predictions - Noise")
    axes[3].set_title("Model Output (Raw & Classified)")
    axes[3].set_xlabel("Time (seconds)")
    axes[3].set_ylabel("Prediction (scaled)")
    axes[3].legend()
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{os.path.splitext(fname)[0]}.png")
    plt.savefig(output_file)
    plt.close()
    print(f"Saved output for {fname} to {output_file}")



def apply_model_to_mssd_compact(mssd_file, model_file="models/ref/cnn_20250613_1327_1922.pt", sampling_rate=50, window_size=100, stride=10, skip_segments=5):

    cfg = Config()
    output_dir = "plots/rs4d_output/"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_file, map_location=torch.device(device))
    fname = os.path.basename(mssd_file)
    try:
        st = read(mssd_file)
    except Exception as e:
        print(f"Error reading {fname}: {e}")
        return
    if len(st) < 3:
        print(f"Skipping {fname}: less than 3 components")
        return
    st.sort(['channel'])
    min_len = min(len(tr.data) for tr in st[:3])
    st = st[:3]
    st = st.copy()
    for i in range(3):
        st[i].data = st[i].data[:min_len]
    data = np.array([tr.data for tr in st[:3]])
    channel_names = [tr.stats.channel for tr in st[:3]]
    original_rate = st[0].stats.sampling_rate
    n_samples = data.shape[1]
    if original_rate != sampling_rate:
        num_samples = int(n_samples * sampling_rate / original_rate)
        data = np.array([resample(ch, num_samples) for ch in data])
    total_samples = data.shape[1]
    total_seconds = total_samples / sampling_rate
    timestamps = np.arange(0, total_seconds, 1 / sampling_rate)
    real_time_window = int(2 * sampling_rate)
    time_preds = []
    raw_output = []
    classified_output = []
    segment_count = 0
    for current_time in range(0, total_samples, stride):
        start = max(0, current_time - real_time_window)
        segment = data[:, start:current_time]
        if segment.shape[1] < window_size:
            continue
        segment_count += 1
        if segment_count <= skip_segments:
            continue
        segment = normalize_data(segment)
        input_tensor = torch.tensor(segment, dtype=torch.float32).unsqueeze(0)
        output = model(input_tensor.to(device))
        if cfg.MODEL_TYPE == cfg.MODEL_TYPE.TFEQ:
            raw_output.append((F.softmax(output, dim=1)[:,1]).item())
            cl_out = output.argmax(dim=1)
        else:
            cl_out = output > 0.57
            raw_output.append(output.item())
        classified_output.append(cl_out.item())
        time_preds.append(current_time / sampling_rate)
    classified_output = np.array(classified_output)
    # Plot all three components in one subplot, model output in another
    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    # Use more distinct and transparent colors
    colors = [(0.2, 0.4, 0.8, 0.7), (0.2, 0.7, 0.2, 0.7), (0.8, 0.2, 0.2, 0.7)]  # blue, green, red with alpha
    for i in range(3):
        axes[0].plot(timestamps, data[i], color=colors[i], label=channel_names[i], alpha=0.7, linewidth=1.5)
    axes[0].set_ylabel("Acceleration (m/s²)",  fontsize=18)
    station_name = st[0].stats.station
    axes[0].set_title(station_name, fontsize=20,  fontweight='bold')
    axes[0].legend()
    axes[1].vlines(time_preds, ymin=0, ymax=raw_output, color="#0651B3", linewidth=2, alpha=0.5)
    #axes[1].plot(time_preds, raw_output, 'o', color="#d62728", markersize=3, label="Model Raw Output", alpha=0.8)
    #axes[1].set_title("Model Raw Output")
    axes[1].set_xlabel("Time (seconds)",  fontsize=18)
    axes[1].set_ylabel("Model Output (RAW)", fontsize=18)
    axes[1].legend()
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{os.path.splitext(fname)[0]}.png")
    plt.savefig(output_file)
    plt.close()
    print(f"Saved compact plot for {fname} to {output_file}")



def download_raspberryshake_data(station, start_time_str, out_dir, fname=None):
    client = Client('RASPISHAKE')
    network = 'AM'
    start_time = UTCDateTime(start_time_str)
    end_time = start_time + 1 * 60  # 3 minutes later
    os.makedirs(out_dir, exist_ok=True)
    try:
        st = client.get_waveforms(network=network, station=station, location="*", channel="*", starttime=start_time, endtime=end_time)
        #st.plot()
        # Remove response and output acceleration
        tr = st[0]
        try:
            inv = client.get_stations(
                network=tr.stats.network,
                station=tr.stats.station,
                location="*",
                channel="*",
                starttime=tr.stats.starttime,
                endtime=tr.stats.endtime,
                level="response"
            )
            print("Inventory contents:", inv.get_contents())
            st.remove_response(inventory=inv, output="ACC", plot=False)
            st.write(fname, format="MSEED")
            print(f"Saved acceleration data to {fname}")
            #apply_model_to_mssd(fname)
            apply_model_to_mssd_compact(fname)

            return fname
        except Exception as e:
            print(f"Failed to remove response for {fname}: {e}")
            print("Trace metadata:", tr.stats)
            return None
    except Exception as e:
        print(f"Failed to download or process data for {station} at {start_time_str}: {e}")
        return None



working = [
    "RD9AE",
    "R04D8",
    "R7734",
    "R4288",
    "R0122",
    "R3AFA"
]
not_working = [
    "RE12C",
    "R5F5E",
    "R71D0"  # Example station
]

for station_name in working:
    download_raspberryshake_data(station_name,"2025-06-05T06:14:23Z","data/rs4d/", fname=f"{station_name}_20250605.mseed")



# cfg = Config()
# nncfg = NNCFG()

# apply_model_to_mssed_files(
#     cfg,
#     mssed_dir='/Users/user/Downloads/mseed/',  # directory with your mssed files
#     output_dir='plots/mssd_output.png',  # where to save the plots
#     model=model,
#     nncfg=nncfg,
#     sampling_rate=50,  # or your model's expected rate
#     window_size=100,   # or your model's expected window size
#     stride=10          # or your preferred stride
# )


# plot_three_components_and_model_output_files(
#     cfg,
#     mssed_dir='/Users/user/Downloads/mseed/',  # directory with your mssed files
#     output_dir='plots/mssd_output_3comp.png',  # where to save the plots
#     model=model,
#     nncfg=nncfg,
#     sampling_rate=50,  # or your model's expected rate
#     window_size=100,   # or your model's expected window size
#     stride=10          # or your preferred stride
# )

#plot_miniseed_directory("/Users/user/Downloads/mseed/")

