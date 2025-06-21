import socket
import numpy as np
import time
from config import Config
import torch
import h5py
from database_op import *

## Network Initialisation
#PI_IP = '192.168.1.8'
PI_IP = '192.168.1.6'
PORT = 65432
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect((PI_IP, PORT))

# data = np.random.rand(10000000).astype(np.float32)
# for i in range(0, len(data), 1000):  # Send 100 samples at a time
#     chunk = data[i:i+100].tobytes()
#     client.sendall(chunk)
#     print(f"Sent chunk {i // 100 + 1}")
#     time.sleep(0.1)  # simulate real-time streaming

cfg = Config()
hdf5_file = "data/waveforms_13_24.hdf5"

# Model parameters
window_size = 1 
stride = 1

with h5py.File(hdf5_file, 'r') as hdf:
    for event_id in hdf["data"].keys():  # Iterate over event IDs
        dataset = hdf["data"].get(event_id)
        if dataset is None:
            continue
        
        start_time = time.time()

        data = np.array(dataset)  # Shape: (num_channels, num_samples)

        # Ensure data shape is correct
        if data.shape[0] != 3:
            print(f"Skipping {event_id}: Expected 3 channels, found {data.shape[0]}")
            continue
        
        # Get total samples and time duration
        total_samples = data.shape[1]
        #total_seconds = total_samples / sampling_rate
        #timestamps = np.arange(0, total_seconds, 1 / sampling_rate)

        chunk_count  = 0

        for start in range(0, total_samples - window_size + 1, stride):
            segment = data[:, start:start + window_size]  # (3, 100)
            ## Transmit the segment
            segment = segment.astype(np.float32)
            #print("segment mean:", np.mean(segment)) 
            chunk = segment.tobytes()
            client.sendall(chunk)
            print(f"Sent - Event ID {event_id}, chunk id {chunk_count}")
            time.sleep(0.018)
            chunk_count +=1

        end_time = time.time()
        print(f"\n\n === Time for one wave {end_time - start_time}")

client.close()
print("Sender finished.")