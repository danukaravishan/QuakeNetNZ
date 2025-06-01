from utils import *
from scipy.signal import detrend, butter, filtfilt
import pywt

def demean(signal):
    """Remove the mean from the signal."""
    return signal - np.mean(signal)

def apply_detrend(signal):
    """Remove linear trends from the signal."""
    return detrend(signal)

def bandpass_filter(signal, lowcut=1, highcut=20.0, fs=50, order=4):
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

def taper_signal(signal, taper_fraction=0.05):
    """Apply a Hann taper to both ends of the signal."""
    npts = len(signal)
    taper_len = int(taper_fraction * npts)
    if taper_len == 0:
        return signal  # Skip if signal too short
    taper = np.ones(npts)
    window = np.hanning(2 * taper_len)
    taper[:taper_len] = window[:taper_len]
    taper[-taper_len:] = window[-taper_len:]
    return signal * taper

def highpass_filter(signal, cutoff=0.5, fs=100.0, order=4):
    """Apply a Butterworth high-pass filter."""
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return filtfilt(b, a, signal)


def normalize(signal):
    """Normalize the signal between -1 and 1."""
    min_val = np.min(signal)
    max_val = np.max(signal)
    if max_val - min_val == 0:
        return np.zeros_like(signal)  # or signal, or any constant value you prefer
    return (signal - min_val) / (max_val - min_val) * 2 - 1
    # """Normalize the signal between -1 and 1."""
    # return (signal - np.min(signal)) / (np.max(signal) - np.min(signal)) * 2 - 1

# normalise 2
def zscore_normalize(signal):
    return (signal - np.mean(signal)) / np.std(signal)

#normalise 3
def minmax_01(signal):
    return (signal - np.min(signal)) / (np.max(signal) - np.min(signal))

# normalise 4
def robust_scale(signal):
    median = np.median(signal)
    iqr = np.percentile(signal, 75) - np.percentile(signal, 25)
    return (signal - median) / iqr

#normalise 5
def energy_normalize(signal):
    energy = np.sqrt(np.sum(signal**2))
    return signal / energy if energy != 0 else signal


def calculate_pga(data):
    norms = np.linalg.norm(data, axis=0)
    pga = np.max(norms)    
    return pga


def wavelet_denoise(x_np, wavelet_name, wavelet_level):
    denoised = []
    for ch in x_np:
        coeffs = pywt.wavedec(ch, wavelet_name, level=wavelet_level)
        coeffs[1:] = [pywt.threshold(c, value=0.2 * np.max(np.abs(c)), mode='soft') for c in coeffs[1:]]
        rec = pywt.waverec(coeffs, wavelet_name)
        rec = rec[:ch.shape[0]]  # Ensure output length matches input
        denoised.append(rec)
    return np.stack(denoised)


def normalize_data(data):
    if data.ndim == 3:  # Case for a set of data
        processed_data = []
        for sig in data:
            normalised = np.array([normalize(sig[i, :]) for i in range(sig.shape[0])])  # Normalize each component
            processed_data.append(normalised)
        return np.array(processed_data)
    elif data.ndim == 2:  # Case for a single input
        return np.array([normalize(data[i, :]) for i in range(data.shape[0])])  # Normalize each component
    else:
        raise ValueError("Input data must have 2 or 3 dimensions.")


def augment_waveform(waveform, amp_factor=2, noise_std=0.0001):
    augmented = waveform.copy()
    channels = [0, 1, 2]
    # Randomly pick two different channels
    ch_up, ch_down = np.random.choice(channels, 2, replace=False)
    # Random factors
    up_factor = 1 + np.random.uniform(0, amp_factor)
    down_factor = 1 - np.random.uniform(0, amp_factor)
    augmented[ch_up, :] *= up_factor
    augmented[ch_down, :] *= down_factor
    return augmented


def clean_low_pga(data, threshold=0.001):
    if data.ndim == 3:  # Case for a set of data
        processed_data = []
        for sig in data:
            pga = float(calculate_pga(sig))*1000
            if pga < threshold:  # Threshold for PGA
                continue
            sigp = augment_waveform(sig)
            processed_data.append(sigp)
        return np.array(processed_data)
    elif data.ndim == 2:  # Case for a single input
        return data if calculate_pga(data) < threshold else [] 
    else:
        raise ValueError("Input data must have 2 or 3 dimensions.")
    

def apply_wavelet_denoise(waveforms, wavelet_name, wavelet_level):

    if waveforms.ndim == 2:
        # Single waveform
        return wavelet_denoise(waveforms, wavelet_name, wavelet_level)
    elif waveforms.ndim == 3:
        # Batch of waveforms
        return np.stack([
            wavelet_denoise(wf, wavelet_name, wavelet_level) for wf in waveforms
        ])
    else:
        raise ValueError("Input must be of shape (3, 100) or (N, 3, 100)")
    
    
def pre_process_real_time_2s(signal, sampling_rate=50):
    processed_data = []
    for sig in signal:
        filtered = bandpass_filter(sig, fs=sampling_rate)
        processed_data.append(filtered)
    return np.array(processed_data)


def pre_proc_data(data, sampling_rate=50):
    processed_data = []
    for sig in data:
        demeaned = demean(sig)
        detrended = apply_detrend(demeaned)
        filtered = bandpass_filter(detrended, fs=sampling_rate)
        #normalized = normalize(filtered)
        processed_data.append(filtered)
    return np.array(processed_data)


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


