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
from report import count_parameters


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
            FP +=1
        else:
            FN +=1
    
    print(f"TP = {TP}, FP = {FP}, TN = {TN}, FN = {FN}")

    # Calculate Accuracy, Precision, Recall, and F1 Score
    accuracy = 100*((TP + TN) / (TP + TN + FP + FN))
    precision = 100*(TP / (TP + FP)) if (TP + FP) != 0 else 0
    recall = 100*(TP / (TP + FN)) if (TP + FN) != 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    parameters = count_parameters(model)

    print(f'Accuracy: {accuracy:.4f}%')
    print(f'Precision: {precision:.4f}%')
    print(f'Recall: {recall:.4f}%')
    print(f'F1 Score: {f1:.4f}%')
    print(f'Parameters: {parameters}')

    file_exists = os.path.isfile(cfg.EQTEST_MODEL_CSV)

    with open(cfg.EQTEST_MODEL_CSV, mode='a', newline='') as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow(['Model Name', 'Input window', 'Shift', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'Model Parameters'])

        writer.writerow(["CRED", f"{cfg.TRAINING_WINDOW}", f"{cfg.SHIFT_WINDOW}", f"{accuracy:.4f}%",f"{precision:.4f}%", f"{recall:.4f}%", f"{f1:.4f}%", parameters])

    print(f"Model details for CRED with input {cfg.TRAINING_WINDOW} and shift from p pick {cfg.SHIFT_WINDOW} appended to {cfg.EQTEST_MODEL_CSV} CSV.")

    return 0


if __name__ == "__main__":
    #main()
    cfg = Config()
    dev(cfg)
    #test()
    #test_cred()
