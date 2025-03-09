from utils import *
from scipy.signal import detrend, butter, filtfilt


def demean(signal):
    """Remove the mean from the signal."""
    return signal - np.mean(signal)

def apply_detrend(signal):
    """Remove linear trends from the signal."""
    return detrend(signal)

def bandpass_filter(signal, lowcut=1.0, highcut=20.0, fs=100, order=4):
    """
    Apply a Butterworth bandpass filter.
    
    Parameters:
    - signal: Input time series
    - lowcut: Low cutoff frequency (Hz)
    - highcut: High cutoff frequency (Hz)
    - fs: Sampling frequency (Hz)
    - order: Filter order
    """
    nyquist = 0.5 * fs  # Nyquist frequency
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def normalize(signal):
    """Normalize the signal between -1 and 1."""
    return (signal - np.min(signal)) / (np.max(signal) - np.min(signal)) * 2 - 1

def pre_proc_data(data):
    return np.array([normalize(bandpass_filter(apply_detrend(demean(sig)))) for sig in data])


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