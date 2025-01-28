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
from extract_window_db import extract_data

seisbench.use_backup_repository() # Since the model weights are downloaded from TCP:2888, which is blocked by crisislab02

# def data():
#     client = Client("GFZ")
#     t = UTCDateTime("2007/01/02 05:49:0")
#     stream = client.get_waveforms(network="CX", station="PB01", location="*", channel="HH?", starttime=t-28.5, endtime=t+1.5)

#     fig = plt.figure(figsize=(15, 5))
#     ax = fig.add_subplot(111)
#     for i in range(3):
#         ax.plot(stream[i].times(), stream[i].data, label=stream[i].stats.channel)
#     ax.legend()
#     return stream


#def main():
    #detections = model.classify(stream)
    #classification = model.classify_aggregate(stream,{"detection_threshold":0.1})
    #classification = model.classify(stream)


def eval(model, cfg):

    hdf5_file = h5py.File(cfg.DATA_EXTRACTED_FILE, 'r')
    p_data_st, noise_data_st = getStreamListFromDatabase(hdf5_file)
    #model = sbm.PhaseNet.from_pretrained("original", update=True)
  
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

        for trace in stream:
            print(f"Trace {trace.id}, samples:= {len(trace.data)}, sampling rate = {trace.stats.sampling_rate}, start time = {trace.stats.starttime}, end time = {trace.stats.endtime}")

        if len(classification.picks) > 0:
            TP +=1
        else:
            FN +=1

    for x in noise_data_st:
        stream = x
        stream = stream.resample(100.0)
        stream.detrend('demean')
        classification = model.classify(stream)
        if len(classification.picks) > 0:
            FP +=1
        else:
            TN +=1
    
    return TP, FP, TN, FN


def exec():
    cfg = Config()
    cfg.argParser()
    extract_data(cfg)

    model = sbm.PhaseNet.from_pretrained("original", update=True)
    #model.cuda()
    TP, FP, TN, FN = eval(model, cfg)

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
    exec()