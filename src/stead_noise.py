# reading the csv file into a dataframe:
import pandas as pd
import matplotlib.pyplot as plt
import obspy
import h5py
from obspy import UTCDateTime
import numpy as np
from obspy.clients.fdsn.client import Client



def make_stream(dataset):
    '''
    input: hdf5 dataset
    output: obspy stream

    '''
    data = np.array(dataset)

    tr_E = obspy.Trace(data=data[:, 0])
    tr_E.stats.starttime = UTCDateTime(dataset.attrs['trace_start_time'])
    tr_E.stats.delta = 0.01
    tr_E.stats.channel = dataset.attrs['receiver_type']+'E'
    tr_E.stats.station = dataset.attrs['receiver_code']
    tr_E.stats.network = dataset.attrs['network_code']

    tr_N = obspy.Trace(data=data[:, 1])
    tr_N.stats.starttime = UTCDateTime(dataset.attrs['trace_start_time'])
    tr_N.stats.delta = 0.01
    tr_N.stats.channel = dataset.attrs['receiver_type']+'N'
    tr_N.stats.station = dataset.attrs['receiver_code']
    tr_N.stats.network = dataset.attrs['network_code']

    tr_Z = obspy.Trace(data=data[:, 2])
    tr_Z.stats.starttime = UTCDateTime(dataset.attrs['trace_start_time'])
    tr_Z.stats.delta = 0.01
    tr_Z.stats.channel = dataset.attrs['receiver_type']+'Z'
    tr_Z.stats.station = dataset.attrs['receiver_code']
    tr_Z.stats.network = dataset.attrs['network_code']

    stream = obspy.Stream([tr_E, tr_N, tr_Z])

    return stream
 

def make_plot(tr, title='', ylab=''):
    '''
    input: trace
    
    '''
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(tr.times("matplotlib"), tr.data, "k-")
    ax.xaxis_date()
    fig.autofmt_xdate()
    plt.ylabel('counts')
    plt.title('Raw Data')
    plt.show()
        

def get_acc_response(dataset, client):

    st = make_stream(dataset)

    start_time = st[0].stats.starttime
    st.trim(start_time+2, start_time + 4)

    try:
    # downloading the instrument response of the station from IRIS
        inventory = client.get_stations(network=dataset.attrs['network_code'],
                                        station=dataset.attrs['receiver_code'],
                                        starttime=UTCDateTime(dataset.attrs['trace_start_time']),
                                        endtime=UTCDateTime(dataset.attrs['trace_start_time']) + 2,
                                        loc="*", 
                                        channel="*",
                                        level="response")  

        st.remove_response(inventory=inventory, output="ACC", plot=False) 
        return st, True
    except:
        return st, False
    
    

def get_noise_samples(sample_count):

    samples = []

    csv_file = "/Users/user/Downloads/chunk1/chunk1.csv"
    file_name = "/Users/user/Downloads/chunk1/chunk1.hdf5"
    dtfl = h5py.File(file_name, 'r')

    client = Client("IRIS")

    df = pd.read_csv(csv_file)
    df = df[df.trace_category == 'noise']
    df_sampled = df.sample(n=sample_count, random_state=42) 
    trace_names = df_sampled['trace_name'].tolist()

    iter_count = 0
    event_list = []
    # Read waveform from HDF5
    for event in trace_names:
        print(f"{iter_count}, Event {event}")
        iter_count+=1

        dataset = dtfl.get('data/' + str(event))

        acc_st, val =  get_acc_response(dataset, client)
        if val:
            acc_st.resample(50)
            acc_data = np.stack([tr.data for tr in acc_st])
            samples.append(acc_data)
            event_list.append(event)
    
    samples_array = np.array(samples)
    event_list_arr = np.array(event_list)
    np.save(f"stead_samples_{sample_count}.npy", samples_array)
    np.save(f"stead_samples_events_{sample_count}.npy", event_list)


get_noise_samples(80000)