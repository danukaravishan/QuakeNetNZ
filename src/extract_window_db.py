# This script configures the hdf5 file required for training.
# This can run standalone with the configurations.

from utils import *
from config import *

def extract_wave_window(data, wave_index, window_size):
    half_window = window_size // 2 
    start_index = max(0, wave_index - half_window)
    end_index = wave_index + half_window    
    end_index = min(end_index, data.shape[1])
    return data[:, start_index:end_index]


def extract_30s_wave_window(data, wave_index, shift ,sampling_rate):
    # Define the fixed window size as 30 seconds' worth of samples
    window_size = 30 * sampling_rate
    wave_index = int(wave_index + shift*sampling_rate)

    start_index = wave_index - window_size  # Left-anchored
    
    # If start_index is negative, calculate how much padding is needed
    if start_index < 0:
        padding_size = abs(start_index)
        # Generate Gaussian noise to pad the beginning
        padding = np.random.normal(0, .000001, (data.shape[0], padding_size))
        # Extract available data from start (0) to wave_index
        extracted_data = data[:, :wave_index]
        # Concatenate the noise with the available data
        window = np.concatenate((padding, extracted_data), axis=1)
    else:
        # If enough data is available, extract directly
        window = data[:, start_index:wave_index]
    
    # Ensure the output has the exact window size (30 seconds' worth of samples)
    if window.shape[1] < window_size:
        # Pad with additional Gaussian noise if the window is still short
        padding = np.random.normal(0, 1, (data.shape[0], window_size - window.shape[1]))
        window = np.concatenate((padding, window), axis=1)
    return window


def extract_noise_window(data, window_size, p_index):
    if p_index >= 200: # Default - Extract the noise window from the begining of the wave
        return data[:,:min(window_size, data.shape[1])]
    else: # Get the noise from the back assuming s wave is not towards the end
        return data[:, -window_size:]

def downsample(data, original_rate, target_rate):
    downsample_factor = int(target_rate // original_rate)
    return decimate(data, downsample_factor, axis=1, zero_phase=True)


def extract_data(cfg=None):

    if cfg is None:
        cfg = Config()

    hdf5_file = h5py.File(cfg.ORIGINAL_DB_FILE, 'r')

    if os.path.isfile(cfg.DATA_EXTRACTED_FILE):
        os.remove(cfg.DATA_EXTRACTED_FILE)

    with h5py.File(cfg.DATA_EXTRACTED_FILE, 'a') as hdf:
        
        # Create database groups
        if 'positive_samples_p' not in hdf:
            positive_group_p = hdf.create_group('positive_samples_p')
        else:
            positive_group_p = hdf['positive_samples_p']

        #if "positive_samples_s" not in hdf:
        #    positive_group_s = hdf.create_group('positive_samples_s')
        #else:
        #    positive_group_s = hdf['positive_samples_s']

        if "negative_sample_group" not in hdf:
            negative_group = hdf.create_group('negative_sample_group')
        else:
            negative_group = hdf['negative_sample_group']

        count = 0
        downsample_factor = 1

        split_index = int(0.8 * len(hdf5_file.keys()))
        
        for event_id in hdf5_file.keys():
            #print(event_id)

            if (count < split_index):
                count +=1
                continue

            dataset = hdf5_file.get(event_id)
            data = np.array(dataset)

            p_arrival_index = int(dataset.attrs["p_arrival_sample"])
            s_arrival_index = int(dataset.attrs["s_arrival_sample"])

            sampling_rate = int(dataset.attrs["sampling_rate"])

            p_wave_picktime = UTCDateTime(dataset.attrs["p_wave_picktime"])
            s_wave_picktime = UTCDateTime(dataset.attrs["s_wave_picktime"])

            wave_time_diff = s_wave_picktime - p_wave_picktime

            if wave_time_diff < 0.2:
                continue

            if sampling_rate != cfg.BASE_SAMPLING_RATE:
                # Add code to resample to 50
                #print(sampling_rate) 
                data_resampled = downsample(data, cfg.BASE_SAMPLING_RATE, sampling_rate)
                data = data_resampled
                downsample_factor = int(sampling_rate // cfg.BASE_SAMPLING_RATE)
                p_arrival_index = int(p_arrival_index/downsample_factor)
                s_arrival_index = int(s_arrival_index/downsample_factor)
                sampling_rate = cfg.BASE_SAMPLING_RATE
            
            count +=1
            
            window_size = int(cfg.TRAINING_WINDOW * sampling_rate)
            # p_data  = extract_wave_window(data, p_arrival_index, window_size)
            # s_data = extract_wave_window(data, s_arrival_index, window_size)
            # noise_data = extract_noise_window(data, window_size, p_arrival_index)
            
            p_data  = extract_30s_wave_window(data, p_arrival_index, cfg.SHIFT_WINDOW, sampling_rate)
            noise_data = extract_30s_wave_window(data, p_arrival_index, -5, sampling_rate)

            if ((len(p_data[0]) != window_size) or (len(noise_data[0]) != window_size)):
                print("Wrong data  ====== : "+event_id)
                continue

            ## Add data to each groups
            if event_id not in positive_group_p:
                positive_p_dataset = positive_group_p.create_dataset(event_id, data=p_data)
            else:
                print(f"Dataset {event_id} already exists in positive_samples_p. Skipping.")

            #if event_id not in positive_group_s:
            #    positive_s_dataset = positive_group_s.create_dataset(event_id, data=s_data)
            #else:
            #    print(f"Dataset {event_id} already exists in positive_samples_s. Skipping.")

            if event_id not in negative_group:
                negative_dataset = negative_group.create_dataset(event_id, data=noise_data)
            else:
                print(f"Dataset {event_id} already exists in negative_group. Skipping.")

            for key, value in dataset.attrs.items():
                positive_group_p[event_id].attrs[key] = value
                #positive_group_s[event_id].attrs[key] = value
                negative_group[event_id].attrs[key] = value
                # Change the wave start time, samppling rate and other changed attributes

            ## Tempory check
            if count > split_index + 100:
                break

    print ("Number of records " + str(count))
