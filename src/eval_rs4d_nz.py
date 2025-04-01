#mseed_file = "data/RS4D/2024p986451stationRD9AE.mseed"
#2024p954889stationR4D20.mseed
#2020p021484stationR4D20
#2023p296804stationRE583.mseed

import torch
import obspy
import numpy as np
import matplotlib.pyplot as plt
from dataprep import pre_proc_data

# Load the pre-trained TorchScript model
model = torch.jit.load("models/quakenetnz.pt")
model.eval()

# Load the Raspberry Shake data from an .mseed file
mseed_file = "data/RS4D/2024p306514stationRE158.mseed"
stream = obspy.read(mseed_file)

target_sampling_rate = 50
for tr in stream:
    tr.resample(target_sampling_rate)

# Ensure all traces have the same length
min_length = min(len(tr.data) for tr in stream)  # Find the shortest trace

# Convert to NumPy array (trimmed to min_length)
traces = np.array([tr.data[:min_length] for tr in stream])  # Shape: (3, min_length)

# Get sampling rate and total duration
sampling_rate = stream[0].stats.sampling_rate  
total_seconds = min_length / sampling_rate  # Total duration in seconds

# Define window and stride
window_size = 100  # Model input size
stride = 10      # Step size

# Apply sliding window
total_samples = traces.shape[1]
#traces = pre_proc_data(traces)
outputs = []  # Store model outputs
timestamps = []  # Store corresponding timestamps

for start in range(0, total_samples - window_size + 1, stride):
    # Extract a segment
    segment = traces[:, start:start + window_size]  # Shape: (3, 100)
    segment = pre_proc_data(segment)

    # Convert to PyTorch tensor and add batch dimension
    input_tensor = torch.tensor(segment, dtype=torch.float32).unsqueeze(0)  # Shape: (1, 3, 100)

    # Run inference
    output = model(input_tensor)  # Get predictions
    predicted_class = torch.argmax(output, dim=1).item()  # Convert to scalar

    # Store result
    outputs.append(1 - predicted_class)
    timestamps.append(start / sampling_rate)  # Convert sample index to time (seconds)

# Convert outputs to NumPy array
outputs = np.array(outputs)  # Shape: (num_windows,)

# PLOTTING
fig, axes = plt.subplots(4, 1, figsize=(12, 6), sharex=True)

# Plot the waveform with all three components in one plot
time_axis = np.linspace(0, total_seconds, min_length)  # Time in seconds

colors = ["g", "r"]
labels = ["N Component", "E Component"]

axes[0].plot(time_axis, traces[0], color="b", label="Z Component")
axes[1].plot(time_axis, traces[1], color="b", label="N Component")
axes[2].plot(time_axis, traces[2], color="b", label="E Component")


# Normalize prediction values for better visualization
normalized_outputs = outputs / np.max(outputs) * np.max(traces)  # Scale to waveform range

# Plot model predictions
axes[3].plot(timestamps, normalized_outputs, color="k", label="Model Predictions", linestyle="dashed", marker="o")
axes[3].legend()
axes[3].set_xlabel("Time (seconds)")
axes[3].set_ylabel("Prediction (scaled)")
axes[3].set_title("Model Predictions Over Time")

plt.suptitle("Waveform and QuakeNetNZ Predictions")
plt.show()
