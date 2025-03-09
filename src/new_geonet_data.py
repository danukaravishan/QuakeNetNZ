# This script configures the hdf5 file required for training.
# This can run standalone with the configurations.

from utils import *
from config import *
import pandas as pd

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

    # db_path = "E:\GeoNet_Earthquake_Dataset\waveforms_units_2013.h5"
    # csv_path = r"E:\GeoNet_Earthquake_Dataset\full_metadata.csv"

    db_path = r"D:\Temp\waveforms_units_2013.h5"
    csv_path = r"D:\Temp\full_metadata.csv"

    metadata_df = pd.read_csv(csv_path)
    
    hdf5_file = h5py.File(db_path, 'r')

    if os.path.isfile(cfg.DATA_EXTRACTED_FILE):
        os.remove(cfg.DATA_EXTRACTED_FILE)

    with h5py.File(cfg.DATA_EXTRACTED_FILE, 'a') as hdf:
        
        # Create database groups
        if 'positive_samples_p' not in hdf:
            positive_group_p = hdf.create_group('positive_samples_p')
        else:
            positive_group_p = hdf['positive_samples_p']

        if "positive_samples_s" not in hdf:
            positive_group_s = hdf.create_group('positive_samples_s')
        else:
            positive_group_s = hdf['positive_samples_s']

        if "negative_sample_group" not in hdf:
            negative_group = hdf.create_group('negative_sample_group')
        else:
            negative_group = hdf['negative_sample_group']

        count = 0
        downsample_factor = 1

        #split_index = int(0.8 * len(hdf5_file.keys()))
        data_group = hdf5_file["data"]
        for trace_name in data_group.keys():
            print(str(count) + "  " + trace_name)

            #if (count < split_index):
            #    count +=1
            #    continue

            dataset = data_group.get(trace_name)
            data = np.array(dataset)

            event_metadata = metadata_df[metadata_df["trace_name"] == trace_name]

            if event_metadata.empty:
                print("No event metadata is available")
                count+=1
                continue

            p_arrival_index = event_metadata["trace_p_arrival_sample"].values[0]
            if p_arrival_index:
                p_arrival_index = int(p_arrival_index) if pd.notna(p_arrival_index) else None
            else:
                print(f" {trace_name} Skipped due to unavailability of P pick time")
                count+=1
                continue

            s_arrival_index = event_metadata["trace_s_arrival_sample"].values[0]

            if s_arrival_index:
                s_arrival_index = int(s_arrival_index) if pd.notna(s_arrival_index) else None
            else:
                print(f" {trace_name} Skipped due to unavailability of S pick time")
                count+=1
                continue

            if p_arrival_index is None or s_arrival_index is None:
                print(f" {trace_name} Skipped due to unavailability of P/S pick time")
                count+=1
                continue
            
            sampling_rate = 100

            #p_wave_picktime = UTCDateTime(event_metadata["trace_p_arrival_time"].values[0])
            #s_wave_picktime = UTCDateTime(event_metadata["trace_s_arrival_time"].values[0])

            #wave_time_diff = s_wave_picktime - p_wave_picktime

            #if wave_time_diff < 0.2:
            #    continue

            # if sampling_rate != cfg.BASE_SAMPLING_RATE:
            #     # Add code to resample to 50
            #     #print(sampling_rate) 
            #     data_resampled = downsample(data, cfg.BASE_SAMPLING_RATE, sampling_rate)
            #     data = data_resampled
            #     downsample_factor = int(sampling_rate // cfg.BASE_SAMPLING_RATE)
            #     p_arrival_index = int(p_arrival_index/downsample_factor)
            #     s_arrival_index = int(s_arrival_index/downsample_factor)
            #     sampling_rate = cfg.BASE_SAMPLING_RATE
            
            count +=1
            
            window_size = int(cfg.TRAINING_WINDOW * sampling_rate)
            p_data  = extract_wave_window(data, p_arrival_index, window_size)
            s_data = extract_wave_window(data, s_arrival_index, window_size)
            noise_data = extract_noise_window(data, window_size, p_arrival_index)
            
            # Enable below two lines to create 30 second shift databases
            #p_data  = extract_30s_wave_window(data, p_arrival_index, cfg.SHIFT_WINDOW, sampling_rate)
            #noise_data = extract_30s_wave_window(data, p_arrival_index, -5, sampling_rate)

            if ( (len(p_data[0]) != window_size) or (len(s_data[0]) != window_size) or (len(noise_data[0]) != window_size)):
                print("Wrong data  ====== : "+trace_name)
                continue

            ## Add data to each groups
            if trace_name not in positive_group_p:
                positive_p_dataset = positive_group_p.create_dataset(trace_name, data=p_data)
            else:
                print(f"Dataset {trace_name} already exists in positive_samples_p. Skipping.")

            if trace_name not in positive_group_s:
                positive_s_dataset = positive_group_s.create_dataset(trace_name, data=s_data)
            else:
                print(f"Dataset {trace_name} already exists in positive_samples_p. Skipping.")

            #if event_id not in positive_group_s:
            #    positive_s_dataset = positive_group_s.create_dataset(event_id, data=s_data)
            #else:
            #    print(f"Dataset {event_id} already exists in positive_samples_s. Skipping.")

            if trace_name not in negative_group:
                negative_dataset = negative_group.create_dataset(trace_name, data=noise_data)
            else:
                print(f"Dataset {trace_name} already exists in negative_group. Skipping.")

            for key, value in dataset.attrs.items():
                positive_group_p[trace_name].attrs[key] = value
                #positive_group_s[event_id].attrs[key] = value
                negative_group[trace_name].attrs[key] = value
                # Change the wave start time, samppling rate and other changed attributes

            ## Testing only
            #if count > split_index + 100:
            #    break

    print ("Number of records " + str(count))

if __name__ == "__main__":
    extract_data()