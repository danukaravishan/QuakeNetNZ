import numpy as np
import torch
from obspy import Stream
from obspy.core import UTCDateTime
from scipy.signal import resample

from obspy.clients.fdsn import Client
from obspy import UTCDateTime
import matplotlib.pyplot as plt
import seisbench.models as sbm


# Sliding window parameters
window_size = 100  # 100 samples per window
stride = 50  # Overlapping step

def preprocess_stream(stream: Stream, target_sampling_rate=100):

    stream.merge(method=1, fill_value="interpolate")
    stream.sort(["channel"])

    # Resample if needed
    for trace in stream:
        if trace.stats.sampling_rate != target_sampling_rate:
            num_samples = int(len(trace.data) * (target_sampling_rate / trace.stats.sampling_rate))
            trace.data = resample(trace.data, num_samples)
            trace.stats.sampling_rate = target_sampling_rate
    
    return stream

def stream_to_sliding_windows(stream: Stream, window_size, stride):
    if len(stream) != 3:
        raise ValueError("Stream must have 3 components (Z, N, E)")

    data = np.vstack([trace.data for trace in stream])
    num_windows = (data.shape[1] - window_size) // stride + 1
    windows = np.array([data[:, i:i + window_size] for i in range(0, num_windows * stride, stride)])
    return windows

def apply_quakenet(stream: Stream, model_path="models/cnn_final.pt"):
    stream = preprocess_stream(stream)
    windows = stream_to_sliding_windows(stream, window_size, stride)
    data_tensor = torch.tensor(windows, dtype=torch.float32)

    model = torch.jit.load(model_path)  # Assuming it's a TorchScript model
    model.eval()

    with torch.no_grad():
        predictions = model(data_tensor)

    return predictions


client = Client("GEONET")
time = UTCDateTime("2024-05-03T22:05:30")
stream = client.get_waveforms(network="NZ", station="WEL", location="*", channel="HH?", starttime=time-100, endtime=time+100)
results = apply_quakenet(stream)
print(results)