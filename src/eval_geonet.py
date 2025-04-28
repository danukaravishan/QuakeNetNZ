import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt
from obspy.signal.filter import bandpass
from dataprep import pre_proc_data
from config import Config
from database_op import *
from scipy.signal import resample

def plot_comparison(segment, test_sample, type1, type2):
    plt.figure(figsize=(12, 4))
    
    for i in range(3):  # Loop over 3 channels
        plt.subplot(1, 3, i + 1)
        plt.plot(segment[i], label=type1)
        plt.plot(test_sample[i], label=type2, linestyle="dashed")
        plt.title(f"{type1} vs {type2} - Channel {i+1}")
        plt.legend()
    
    plt.show()


# Load the pre-trained TorchScript model
cfg = Config()

model = torch.jit.load(cfg.MODEL_FILE_NAME)
model.eval()

# Path to HDF5 file
hdf5_file = "/Users/user/Library/CloudStorage/OneDrive-MasseyUniversity/Technical-Work/databackup/waveforms.hdf5"

# Model parameters
window_size = 100  # Model input size
stride = 10  # Step size
sampling_rate = 50  # Ensure consistency with training data

cfg = Config()

hdf5_file1 = h5py.File(cfg.TEST_DATA, 'r')
p_data, s_data, noise_data = getWaveData(cfg, hdf5_file1)

p_data      = np.array(p_data)
s_data      = np.array(s_data)
noise_data  = np.array(noise_data)

p_data = pre_proc_data(p_data)
s_data = pre_proc_data(s_data)
noise_data = pre_proc_data(noise_data)

nncfg = NNCFG()

# Open HDF5 file and iterate through all waveforms
with h5py.File(hdf5_file, 'r') as hdf:
    for event_id in reversed(list(hdf.keys())):  # Iterate over event IDs
        dataset = hdf.get(event_id)
        if dataset is None:
            continue
        
        data = np.array(dataset)  # Shape: (num_channels, num_samples)

        # Ensure data shape is correct
        if data.shape[0] != 3:
            print(f"Skipping {event_id}: Expected 3 channels, found {data.shape[0]}")
            continue
        
        if dataset.attrs["sampling_rate"]  != sampling_rate:
            # Resample the data to the target sampling rate
            original_rate = dataset.attrs["sampling_rate"]
            num_samples = int(data.shape[1] * sampling_rate / original_rate)
            data = np.array([resample(channel, num_samples) for channel in data])
    
        # Get total samples and time duration
        total_samples = data.shape[1]
        total_seconds = total_samples / sampling_rate
        timestamps = np.arange(0, total_seconds, 1 / sampling_rate)

        # Apply sliding window inference
        time_preds = []

        raw_output = []
        classified_output = []


        for start in range(0, total_samples - window_size + 1, stride):
            segment = data[:, start:start + window_size]  # (3, 100)
            segment = pre_proc_data(segment)
            input_tensor = torch.tensor(segment, dtype=torch.float32).unsqueeze(0)  # (1, 3, 100)
            output = model(input_tensor)

            cl_out = output > 0.99

            #predicted_class = torch.argmax(output, dim=1).item()

            classified_output.append(cl_out.item())  # Negate the class output
            raw_output.append(output.item())
            time_preds.append(start / sampling_rate)

        # Convert to NumPy array
        classified_output = np.array(classified_output)

        # PLOTTING
        fig, axes = plt.subplots(3, 1, figsize=(12, 6), sharex=True)

        # Plot the waveform
        colors = ["b", "g", "r"]
        labels = ["Z Component", "N Component", "E Component"]
        for i in range(3):
            axes[0].plot(timestamps, data[i], color=colors[i], label=labels[i])

        axes[0].legend()
        axes[0].set_ylabel("Amplitude")
        axes[0].set_title(f"Seismic Waveform - Event {event_id}")

        # Normalize predictions to match waveform scale
        #normalized_outputs = (outputs - np.min(outputs)) / (np.max(outputs) - np.min(outputs))

        # Plot model predictions
        #axes[1].plot(time_preds, normalized_outputs, color="k", label="Model Predictions", linestyle="dashed", marker="o")
        
        axes[1].vlines(time_preds, ymin=0, ymax=classified_output, color="k", linewidth=1)
        axes[1].plot(time_preds, classified_output, 'o', color="k", markersize=3, label="Model Predictions- noise")

        axes[1].legend()
        axes[1].set_xlabel("Time (seconds)")
        axes[1].set_ylabel("Prediction (scaled)")
        axes[1].set_title("Model Predictions Over Time")

        axes[2].vlines(time_preds, ymin=0, ymax=raw_output, color="k", linewidth=1)
        axes[2].plot(time_preds, raw_output, 'o', color="r", markersize=3, label="Model Predictions- earthquake")

        plt.suptitle(f"Waveform and QuakeNetNZ Predictions - Event {event_id}")
        plt.show()

        # Keyboard input to continue or exit
        user_input = input("Press Enter to continue or type 'q' to quit: ")
        if user_input.lower() == 'q':
            print("Exiting loop.")
            break  # Exit the loop if user types 'q'
