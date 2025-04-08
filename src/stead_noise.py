# reading the csv file into a dataframe:
import pandas as pd
import h5py
import numpy as np
import matplotlib.pyplot as plt

import obspy
import h5py
from obspy import UTCDateTime
import numpy as np
from obspy.clients.fdsn.client import Client
import matplotlib.pyplot as plt
import random


selected_traces = set()
inventory_cache = {}

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
    



def get_acc_response_opt(dataset):
    global inventory_cache

    st = make_stream(dataset)

    key = (dataset.attrs['network_code'], dataset.attrs['receiver_code'])
    start_time = UTCDateTime(dataset.attrs['trace_start_time'])

    try:
        # Use cached inventory if available
        if key not in inventory_cache:
            client = Client("IRIS")
            inventory = client.get_stations(network=key[0],
                                            station=key[1],
                                            starttime=start_time,
                                            endtime=start_time + 60,
                                            loc="*", 
                                            channel="*",
                                            level="response")
            inventory_cache[key] = inventory
        else:
            inventory = inventory_cache[key]

        st.remove_response(inventory=inventory, output="ACC", plot=False)
        return st, True
    
    except Exception as e:
        print(f"Error with {key}: {e}")
        return st, False
    

def get_acc_response(dataset):

    st = make_stream(dataset)

    try:
    # downloading the instrument response of the station from IRIS
        client = Client("IRIS")
        inventory = client.get_stations(network=dataset.attrs['network_code'],
                                        station=dataset.attrs['receiver_code'],
                                        starttime=UTCDateTime(dataset.attrs['trace_start_time']),
                                        endtime=UTCDateTime(dataset.attrs['trace_start_time']) + 60,
                                        loc="*", 
                                        channel="*",
                                        level="response")  


    # # ploting the verical component
    # st = make_stream(dataset)
    # make_plot(st[2], title='Displacement', ylab='meters')

    # st = make_stream(dataset)
    # st = st.remove_response(inventory=inventory, output='VEL', plot=False) 
    # make_plot(st[2], title='Velocity', ylab='meters/second')

    
        st.remove_response(inventory=inventory, output="ACC", plot=False) 
        return st, True
    except:
        return st, False
    


def get_random_stead_noise_sample():

    global selected_traces

    csv_file = "/Users/user/Downloads/chunk1/chunk1.csv"
    file_name = "/Users/user/Downloads/chunk1/chunk1.hdf5"

    # Load and filter the dataframe
    df = pd.read_csv(csv_file)
    #print(f'Total events in csv file: {len(df)}')

    # Filter only noise traces
    df = df[df.trace_category == 'noise']
    #print(f'Total noise events: {len(df)}')

    # Group by receiver_code
    receiver_groups = df.groupby("receiver_code")

    # Keep trying until a valid, not-yet-picked trace is found
    picked_trace = None
    while picked_trace is None:
        # Pick a random receiver_code
        random_receiver = random.choice(list(receiver_groups.groups.keys()))
        group = receiver_groups.get_group(random_receiver)

        # Exclude traces already picked
        available = group[~group['trace_name'].isin(selected_traces)]

        if not available.empty:
            # Randomly select one trace from available ones
            row = available.sample(n=1).iloc[0]
            picked_trace = row['trace_name']
            selected_traces.add(picked_trace)
        else:
            # All traces under this receiver_code are used, try another
            continue

    print(f"Picked trace in STEAD : {picked_trace}")

    # Read waveform from HDF5
    dtfl = h5py.File(file_name, 'r')
    dataset = dtfl.get('data/' + str(picked_trace))

    return  get_acc_response(dataset)
    

def get_random_sample():
    st, val =  get_random_stead_noise_sample()
    while not val:
        st, val =  get_random_stead_noise_sample()
    
    st.resample(50)
    acc_data = np.stack([tr.data for tr in st])
    acc_data = acc_data[:, 100:200]
    a = 0 # Temp
    return acc_data


samples = []

for x in range (60000):
    sample = get_random_sample()
    samples.append(sample)

samples_array = np.array(samples)
np.save("stead_samples.npy", samples_array)