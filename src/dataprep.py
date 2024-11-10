from utils import *

def highpass_filter():
    print("High Pass Filter the data")

def normalize():
    print("Normalise")


def numpy_to_stream(data):
    # Additional information (optional)
    sampling_rate = 50  # Hz
    station_id = "NA"
    network_code = "NZ"
    p_pick_time = obspy.core.utcdatetime.UTCDateTime(data.attrs["p_wave_picktime"])
    start_time = p_pick_time - 2
    
    # Convert numpy array to list of traces, one for each component
    stream = Stream()
    components = ['Z', 'N', 'E']  # Component labels for channel codes
    for i, component in enumerate(components):
        trace = Trace(
            data=data[i, :],  # Each component is one row in data
            header={
                "sampling_rate": sampling_rate,
                "station": station_id,
                "network": network_code,
                "channel": f"{component}",
                "starttime": start_time
            }
        )
        stream.append(trace)

    return stream


def extractStreamDataFromHDF5Group(group):
    st_list = []
    for event_id in group.keys():
        dataset = group[event_id]
        st = numpy_to_stream(dataset)
        st_list.append(st)    # Append to list
    return st_list


# Ignore S wave data.
def getStreamListFromDatabase(hdf5_file):

    positive_group_p = hdf5_file['positive_samples_p']
    #positive_group_s = hdf5_file['positive_samples_s']
    negative_group   = hdf5_file['negative_sample_group']

    p_data = extractStreamDataFromHDF5Group(positive_group_p)
    #s_data = extractStreamDataFromHDF5Group(positive_group_s)
    noise_data= extractStreamDataFromHDF5Group(negative_group)

    hdf5_file.close()

    return p_data, noise_data