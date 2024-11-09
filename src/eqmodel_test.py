#This script needs to be developed to get input from hdf5 database

import obspy
import seisbench
import seisbench.models as sbm
import torch
from obspy.clients.fdsn import Client
from obspy import UTCDateTime
import matplotlib.pyplot as plt

from utils import *
from dataprep import getStreamListFromDatabase
from config import Config

def data():
    # client = Client("GFZ")
    # t = UTCDateTime("2007/01/01 05:48:50")
    # stream = client.get_waveforms(network="CX", station="PB01", location="*", channel="HH?", starttime=t-10, endtime=t+10)

    #Not picking well with the GeoNet Data
    # client = Client("GEONET")
    # time = UTCDateTime("2024-04-06T23:37:00")
    # stream = client.get_waveforms(network="NZ", station="WEL", location="*", channel="HH?", starttime=time-200, endtime=time+200)

    client = Client("GFZ")
    t = UTCDateTime("2007/01/02 05:49:0")
    stream = client.get_waveforms(network="CX", station="PB01", location="*", channel="HH?", starttime=t-28.5, endtime=t+1.5)

    fig = plt.figure(figsize=(15, 5))
    ax = fig.add_subplot(111)
    for i in range(3):
        ax.plot(stream[i].times(), stream[i].data, label=stream[i].stats.channel)
    ax.legend()
    return stream



def main():
    fig = plt.figure(figsize=(15, 5))
    ax = fig.add_subplot(111)

    stream = data()

    for i in range(3):
        ax.plot(stream[i].times(), stream[i].data, label=stream[i].stats.channel)
    ax.legend()

    model = sbm.PhaseNet.from_pretrained("original")
    #model = sbm.EQTransformer(in_channels=3, in_samples=1000, classes=2, phases='PS', lstm_blocks=3, drop_rate=0.1, original_compatible=False, sampling_rate=100)
    #model = sbm.CRED.from_pretrained("original")

    print(model.weights_docstring)
    #detections = model.classify(stream)
    annotations = model.annotate(stream)
    #classification = model.classify_aggregate(stream,{"detection_threshold":0.1})
    classification = model.classify(stream)

    # For PhaseNetclassification.picks
    #if len(classification.detections) >0:
    if len(classification.picks) > 0:
        print("Detected")
    else:
        print("Not detected")
    return 0
    
    # Example 1: Threshold-based detection if output is a score
    threshold = 0.5  # Set your threshold based on experimentation or model documentation
    is_earthquake_detected = any(score > threshold for score in detections)

    # Example 2: Direct label checking if output is labels
    is_earthquake_detected = any(label == 1 for label in detections)  # Assumes 1 indicates earthquake

    # Output the result
    if is_earthquake_detected:
        print("Earthquake detected!")
    else:
        print("No earthquake detected.")
        
    if annotations.count() == 0:
        print("No earthquake signal detected")
        return 0
    
    print("Earthquke Detected")
    print(annotations)

    if annotations.count() == 1:
        annotations.plot()
        return 0
    
    
    fig = plt.figure(figsize=(15, 10))
    axs = fig.subplots(2, 1, sharex=True, gridspec_kw={'hspace': 0})

    offset = annotations[0].stats.starttime - stream[0].stats.starttime
    for i in range(3):
        axs[0].plot(stream[i].times(), stream[i].data, label=stream[i].stats.channel)
        if annotations[i].stats.channel[-1] != "N":  # Do not plot noise curve
            axs[1].plot(annotations[i].times() + offset, annotations[i].data, label=annotations[i].stats.channel)

    axs[0].legend()
    axs[1].legend()
    plt.show()


def test():
    model = sbm.GPD()
    model = sbm.EQTransformer(in_channels=3, in_samples=200, classes=2, phases='PS', lstm_blocks=3, drop_rate=0.1, original_compatible=False, sampling_rate=100)
    #print(model)
    x = torch.rand(1, 3, 200)  # 1 example, 3 components, 3001 samples
    model.eval()
    with torch.no_grad():
        print(model(x))


def dev(cfg):
    hdf5_file = h5py.File(cfg.DATA_EXTRACTED_FILE, 'r')
    p_data_st, noise_data_st = getStreamListFromDatabase(hdf5_file)
    #model = sbm.PhaseNet.from_pretrained("original", update=True)
    model = sbm.CRED.from_pretrained("original", update=True)
    
    ## Accuracy matrix
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    
    for x in p_data_st:
        stream = x
        stream = stream.resample(100.0)
        stream.detrend('demean')
        #annotations = model.annotate(stream)
        #classification = model.classify_aggregate(stream,{"detection_threshold":0.1})
        classification = model.classify(stream)

        if len(classification.detections) > 0:
            #print("Detected")
            TP +=1
        else:
            #print("Not detected")
            TN +=1

    for x in noise_data_st:
        stream = x
        stream = stream.resample(100.0)
        stream.detrend('demean')
        #annotations = model.annotate(stream)
        #classification = model.classify_aggregate(stream,{"detection_threshold":0.1})
        classification = model.classify(stream)

        if len(classification.detections) > 0:
            #print("Detected")
            FP +=1
        else:
            #print("Not detected")
            FN +=1
    
    print(f"TP = {TP}, FP = {FP}, TN = {TN}, FN = {FN}")

    return 0


if __name__ == "__main__":
    #main()
    cfg = Config()
    dev(cfg)
    #test()
    #test_cred()
