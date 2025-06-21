# This script configures the hdf5 file required for training.
# This can run standalone with the configurations.

from utils import *
from config import *
import scipy.signal
from dataprep import pre_proc_data

def generate_noise(noise_type, shape, sampling_rate=50):
    if noise_type == "gaussian":
        return np.random.normal(0, 0.001, shape)  # Mean 0, Std 0.01
    
    elif noise_type == "uniform":
        return np.random.uniform(-0.01, 0.001, shape)  # Uniform noise between -0.01 and 0.01

    elif noise_type == "pink":
        return generate_pink_noise(shape)

    elif noise_type == "brownian":
        return generate_brownian_noise(shape)

    else:
        raise ValueError("Unknown noise type!")

def generate_pink_noise(shape):
    # Create a filter for pink noise
    white_noise = np.random.normal(0, 1, shape)
    b, a = scipy.signal.butter(1, 0.1, 'low')
    return scipy.signal.filtfilt(b, a, white_noise) * 0.01

def generate_brownian_noise(shape):
    noise = np.cumsum(np.random.normal(0, 0.001, shape), axis=1)  # Accumulated random steps 
    return noise



def extract_noise_from_stead():

    chunk_file = "data/STEAD/chunk.hdf5"
    csv_file = "data/STEAD/merged.csv"
    
    df = pd.read_csv(csv_file)
    df = df[(df.trace_category == 'noise') & (df.receiver_code == 'PHOB')]

    # Ensure we have enough samples
    if len(df) < 50000:
        raise ValueError("Not enough noise samples available!")

    # Randomly select 50,000 trace names
    ev_list = df['trace_name'].sample(n=50000, random_state=42).to_list()

    # Open HDF5 file
    dtfl = h5py.File(chunk_file, 'r')

    # Initialize array to store waveforms
    noise_data = np.zeros((50000, 100, 3))

    # Extract data
    for i, evi in enumerate(ev_list):
        dataset = dtfl.get(f'data/{evi}')
        if dataset is None:
            continue  # Skip if dataset is missing

        data = np.array(dataset)  # Shape: (T, 3), where T is time samples

        if data.shape[0] < 100:
            continue  # Skip if not enough samples

        noise_data[i] = data[:100, :]  # Take first 100 samples

    # Close HDF5 file
    dtfl.close()

    # Save the NumPy array (optional)
    np.save("data/STEAD/noise_samples.npy", noise_data)

    # Print shape to verify
    print("Extracted noise data shape:", noise_data.shape)  # Expected: (50000, 100, 3)


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

def upsample(data, original_rate, target_rate):
    upsample_factor = target_rate / original_rate
    num_samples = int(data.shape[1] * upsample_factor)
    return scipy.signal.resample(data, num_samples, axis=1)


def resample_data(data, original_rate, target_rate):
    """
    Resamples the data to the target_rate. Upsamples if target_rate > original_rate,
    downsamples if target_rate < original_rate.
    """
    if target_rate == original_rate:
        return data
    elif target_rate > original_rate:
        upsample_factor = target_rate / original_rate
        num_samples = int(data.shape[1] * upsample_factor)
        return scipy.signal.resample(data, num_samples, axis=1)
    else:
        downsample_factor = int(original_rate // target_rate)
        return scipy.signal.decimate(data, downsample_factor, axis=1, zero_phase=True)
    
# Extract data from STEAD database
def extract_stead_data(cfg=None):

    if cfg is None:
        cfg = Config()

    hdf5_file = h5py.File("/Users/user/Library/CloudStorage/OneDrive-MasseyUniversity/Technical-Work/databackup/stead_acc_p_s_data.hdf5", 'r')
    extracted_file = "data/stead_2s_data.hdf5"
    if os.path.isfile(extracted_file):
        os.remove(extracted_file)

    stead_noise_data = np.load("stead_samples_80000.npy")

    with h5py.File(extracted_file, 'a') as hdf:
        
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
        stead_noise_index = 0

        #split_index = int(0.8 * len(hdf5_file.keys()))
        
        for event_id in hdf5_file["data"].keys():  # Access the "data" group and iterate over its keys

            dataset = hdf5_file["data"].get(event_id)
            data = np.array(dataset)

            p_arrival_index = int(dataset.attrs["p_arrival_sample"])
            s_arrival_index = int(dataset.attrs["s_arrival_sample"])

            sampling_rate = 100 

            count +=1
            
            ## Give temporal shift to the data
            SHIFT_RANGE_SEC = 0.5  # Max shift in seconds (±0.5 sec)
            
            shift_samples = int(SHIFT_RANGE_SEC * sampling_rate)
            random_shifts_p = np.random.randint(-shift_samples, shift_samples + 1)
            random_shifts_s = np.random.randint(-shift_samples, shift_samples + 1)

            # Modify P and S indices
            new_p_indices = np.clip(p_arrival_index + random_shifts_p, 0, len(data[0]) - 1)
            new_s_indices = np.clip(s_arrival_index + random_shifts_s, 0, len(data[0]) - 1)

            # Extract wave data
            window_size = int(cfg.TRAINING_WINDOW * sampling_rate)
            p_data  = extract_wave_window(data, new_p_indices, window_size)
            s_data = extract_wave_window(data, new_s_indices, window_size)
            noise_data = extract_noise_window(data, window_size, p_arrival_index)

            stead_noise_sample = stead_noise_data[stead_noise_index]
            upsample_factor = sampling_rate/cfg.BASE_SAMPLING_RATE  # 100Hz / 50Hz
            stead_noise_sample = scipy.signal.resample(stead_noise_sample, int(stead_noise_sample.shape[1] * upsample_factor), axis=1)
            stead_noise_index  += 1

            NOISE_MEAN = 0  # Gaussian noise mean
            NOISE_STD = 0.00001  # Gaussian noise std deviation
          
            if np.random.choice([True, False]):
                stead_noise_sample += np.random.normal(NOISE_MEAN, NOISE_STD, stead_noise_sample.shape)

            if ( (len(p_data[0]) != window_size) or (len(s_data[0]) != window_size) or (len(noise_data[0]) != window_size) or (len(stead_noise_sample[0]) != window_size)):
                print("Wrong data  ====== : "+event_id)
                continue

            ## Add data to each groups
            if event_id not in positive_group_p:
                positive_p_dataset = positive_group_p.create_dataset(event_id, data=p_data)
            else:
                print(f"Dataset {event_id} already exists in positive_samples_p. Skipping.")

            if event_id not in positive_group_s:
                positive_s_dataset = positive_group_s.create_dataset(event_id, data=s_data)
            else:
                print(f"Dataset {event_id} already exists in positive_samples_p. Skipping.")

            if event_id not in negative_group:
                negative_dataset = negative_group.create_dataset(event_id, data=noise_data)
                negative_dataset = negative_group.create_dataset(event_id+"_stead", data=stead_noise_sample)
            else:
                print(f"Dataset {event_id} already exists in negative_group. Skipping.")

            for key, value in dataset.attrs.items():
                positive_group_p[event_id].attrs[key] = value
                #positive_group_s[event_id].attrs[key] = value
                negative_group[event_id].attrs[key] = value
                # Change the wave start time, samppling rate and other changed attributes

            print(f" {str(count)} : {event_id}")

    print ("Number of records " + str(count))

#  This function will merge new 2 s data to existing waveform_2s_data.hdf5 file.
# Input the file path of the new data
def merge_new_data(file_path, cfg=None):
    cfg = Config()

    print(f"Merge data from the new file {file_path}  to the existing database data_extracted_file")

    try:
        hdf5_file = h5py.File(file_path, 'r')
    except OSError as e:
        print(f"Error opening HDF5 file: {e}")
        return

    data_extracted_file = "data/waveform_2s_data.hdf5"
    try:
        with h5py.File(data_extracted_file, 'a') as hdf:
            pass  # Ensure the file is accessible
    except OSError as e:
        print(f"Error accessing data_extracted_file: {e}")
        return


    with h5py.File(data_extracted_file, 'a') as hdf:
        
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
        
        for event_id in hdf5_file["data"].keys():  # Directly iterate over the keys in the HDF5 file

            dataset = hdf5_file.get(f"data/{event_id}")
            data = np.array(dataset)

            magnitude = dataset.attrs.get("source_magnitude", 0)
            channel_code = dataset.attrs.get("station_channels", "")
            epicentral_distance = dataset.attrs.get("path_ep_distance_km", float("inf"))
            
            ## Data filtering
            if magnitude < 3:
                continue 

            if magnitude < 5 and epicentral_distance > 100:
                continue
            
            if epicentral_distance  > 150:
                continue

            if len(channel_code) <= 2:    
                print(f"Skipping event {event_id} due to missing or empty channel code.")
                continue

            if len(channel_code[0]) > 1 and channel_code[0][1] != "N":
                continue
            
            p_arrival_sample = dataset.attrs["trace_p_arrival_sample"]
            s_arrival_sample = dataset.attrs["trace_s_arrival_sample"]

            p_data_availabe = False
            s_data_availabe = False
            noise_data_availabe = True

            if not np.isnan(p_arrival_sample):
                p_data_availabe = True
            
            if not np.isnan(s_arrival_sample):
                s_data_availabe = True
            
            if not (p_data_availabe or s_data_availabe):
                continue


            sampling_rate = int(dataset.attrs["trace_sampling_rate_Hz"])

            if p_data_availabe:
                p_arrival_index = int(p_arrival_sample)
                p_wave_picktime = UTCDateTime(dataset.attrs["trace_p_arrival_time"])    

            if s_data_availabe:
                s_arrival_index = int(s_arrival_sample)
                s_wave_picktime = UTCDateTime(dataset.attrs["trace_s_arrival_time"])


            if p_data_availabe and s_data_availabe:
                if s_wave_picktime - p_wave_picktime < 0.2:
                    continue

            data_pre_proc = pre_proc_data(data, sampling_rate=sampling_rate)

            if sampling_rate != cfg.BASE_SAMPLING_RATE:
                data_pre_proc = downsample(data_pre_proc, cfg.BASE_SAMPLING_RATE, sampling_rate)
                downsample_factor = int(sampling_rate // cfg.BASE_SAMPLING_RATE)
                p_arrival_index = int(p_arrival_index/downsample_factor) if p_data_availabe else None
                s_arrival_index = int(s_arrival_index/downsample_factor) if s_data_availabe else None
                sampling_rate = cfg.BASE_SAMPLING_RATE
            
            count += 1
            
            ## Give temporal shift to the data
            SHIFT_RANGE_SEC = 0.5  # Max shift in seconds (±0.5 sec)
            
            shift_samples = int(SHIFT_RANGE_SEC * sampling_rate)
            random_shifts_p = np.random.randint(-shift_samples, shift_samples + 1)
            random_shifts_s = np.random.randint(-shift_samples, shift_samples + 1)

            # Modify P and S indices
            new_p_indices = np.clip(p_arrival_index + random_shifts_p, 0, len(data[0]) - 1) if p_data_availabe else None
            new_s_indices = np.clip(s_arrival_index + random_shifts_s, 0, len(data[0]) - 1) if s_data_availabe else None

            # Extract wave data
            window_size = int(cfg.TRAINING_WINDOW * sampling_rate)
            p_data  = extract_wave_window(data_pre_proc, new_p_indices, window_size) if p_data_availabe else None
            s_data = extract_wave_window(data_pre_proc, new_s_indices, window_size)  if s_data_availabe else None
            
            #noise_data = extract_noise_window(data_pre_proc, window_size, 201)
            # Add another noise sample from unfiltered data randomly between the begining and the p pick time
            
            start_index = 0
            end_index_clip_for_noise = p_arrival_index if p_data_availabe else s_arrival_index
            end_index = max(1, end_index_clip_for_noise - window_size)  # Ensure at least one sample is available
            random_index1 = np.random.randint(start_index, end_index)
            random_index2 = np.random.randint(start_index, end_index)

            noise_data = data_pre_proc[:, random_index2:random_index2 + window_size]
            unprocessed_noise_sample = data[:, random_index1:random_index1 + window_size]


            # Ensure the extracted noise sample has the correct shape
            
            if noise_data.shape[1] < window_size:
                padding = np.random.normal(0, 0.00001, (data.shape[0], window_size - noise_data.shape[1]))
                noise_data = np.concatenate((noise_data, padding), axis=1)

            if unprocessed_noise_sample.shape[1] < window_size:
                padding = np.random.normal(0, 0.00001, (data.shape[0], window_size - unprocessed_noise_sample.shape[1]))
                unprocessed_noise_sample = np.concatenate((unprocessed_noise_sample, padding), axis=1)
            

            # Add random gaussian noise to unprocessed noise sample
            if np.random.choice([True, False]):
                NOISE_MEAN = 0  # Gaussian noise mean
                NOISE_STD = 0.00001  # Gaussian noise std deviation          
                unprocessed_noise_sample += np.random.normal(NOISE_MEAN, NOISE_STD, unprocessed_noise_sample.shape)

            if  p_data_availabe and len(p_data[0]) != window_size:
                print("Wrong data  ====== : "+event_id)
                p_data_availabe = False
            if  s_data_availabe and len(s_data[0]) != window_size:
                print("Wrong data  ====== : "+event_id)
                s_arrival_index = False
            if len(noise_data[0]) != window_size:
                print("Wrong data  ====== : "+event_id)
                noise_data_availabe = False
            if len(unprocessed_noise_sample[0]) != window_size:
                print("Wrong data  ====== : "+event_id)
                noise_data_availabe = False

            ## Add data to each groups
            if p_data_availabe and event_id not in positive_group_p:
                positive_p_dataset = positive_group_p.create_dataset(event_id, data=p_data)
            
            if s_data_availabe and event_id not in positive_group_s:
                positive_s_dataset = positive_group_s.create_dataset(event_id, data=s_data)
            
            if event_id not in negative_group and noise_data_availabe:
                negative_dataset = negative_group.create_dataset(event_id, data=noise_data)
                negative_dataset = negative_group.create_dataset(event_id+"_raw", data=unprocessed_noise_sample)
            
            for key, value in dataset.attrs.items():
                if p_data_availabe:
                    positive_group_p[event_id].attrs[key] = value 
                if s_data_availabe:
                    positive_group_s[event_id].attrs[key] = value
                if noise_data_availabe:
                    negative_group[event_id].attrs[key] = value
                # Change the wave start time, samppling rate and other changed attributes

            print(f" {str(count)} : {event_id}")

    print ("Number of records " + str(count))



def extract_p_data_for_new_data(cfg=None):

    print("Extracting P data from the 90  geonet database")
    if cfg is None:
        cfg = Config()

    hdf5_file = h5py.File(cfg.ORIGINAL_DB_FILE, 'r')

    if os.path.isfile(cfg.DATA_EXTRACTED_FILE):
        os.remove(cfg.DATA_EXTRACTED_FILE)

    #stead_noise_data = np.load("stead_samples_80000.npy")

    with h5py.File(cfg.DATA_EXTRACTED_FILE, 'a') as hdf:
        
        # Create database groups
        if 'positive_samples_p' not in hdf:
            positive_group_p = hdf.create_group('positive_samples_p')
        else:
            positive_group_p = hdf['positive_samples_p']

        if "negative_sample_group" not in hdf:
            negative_group = hdf.create_group('negative_sample_group')
        else:
            negative_group = hdf['negative_sample_group']

        count = 0
        non_p_arrival_count = 0
        
        for event_id in hdf5_file["data"].keys():  # Directly iterate over the keys in the HDF5 file

            if "2016p858000" in event_id:
                continue  # Exclude Kaikoura earthquake events
    
            dataset = hdf5_file["data"].get(event_id)
            data = np.array(dataset)

            if data.shape[1] <= 50:
                continue

            p_arrival_sample = dataset.attrs.get("p_arrival_sample", None)
            if p_arrival_sample is None or str(p_arrival_sample).strip() == "" or str(p_arrival_sample).lower() == "nan":
                print(f"Skipping event {event_id}: empty p_arrival_sample ({p_arrival_sample})")
                non_p_arrival_count +=1
                continue
            p_arrival_index = int(float(p_arrival_sample))

            sampling_rate = int(dataset.attrs["sampling_rate"])

            data_pre_proc = pre_proc_data(data, sampling_rate=sampling_rate)

            if sampling_rate != cfg.BASE_SAMPLING_RATE:
                original_length = data_pre_proc.shape[1]
                data_pre_proc = resample_data(data_pre_proc, sampling_rate, cfg.BASE_SAMPLING_RATE)
                new_length = data_pre_proc.shape[1]
                # Adjust indices proportionally for both upsampling and downsampling
                p_arrival_index = int(p_arrival_index * new_length / original_length)
                sampling_rate = cfg.BASE_SAMPLING_RATE
                
            if data_pre_proc.shape[1] <= p_arrival_index:
                continue
            
            if p_arrival_index < 0:
                continue

            ## Give temporal shift to the data
            SHIFT_RANGE_SEC = 0.5  # Max shift in seconds (±0.5 sec)
            
            shift_samples = int(SHIFT_RANGE_SEC * sampling_rate)
            random_shifts_p = np.random.randint(-shift_samples, shift_samples + 1)

            # Modify P and S indices
            new_p_indices = np.clip(p_arrival_index + random_shifts_p, 0, len(data[0]) - 1)

            # Extract wave data
            window_size = int(cfg.TRAINING_WINDOW * sampling_rate)
            p_data  = extract_wave_window(data_pre_proc, new_p_indices, window_size)

            # Add noise samples
            # start_index = 0
            # end_index = max(1, p_arrival_index - window_size)  # Ensure at least one sample is available
            # random_index1 = np.random.randint(start_index, end_index)
            # random_index2 = np.random.randint(start_index, end_index)

            exclude_start = max(0, p_arrival_index - 2*sampling_rate)
            exclude_end = min(data.shape[1], p_arrival_index + 2*sampling_rate)

            # Build a list of valid start indices for the noise window
            valid_indices = list(range(20, data.shape[1] - window_size + 1 - 10)) # Ignore the first 20 samples and last 10 samples due to edge effect of the signals
            valid_indices = [i for i in valid_indices if i + window_size <= exclude_start or i >= exclude_end]

            if len(valid_indices) < 2:
                print(f"Not enough valid noise windows for event {event_id}")
                continue

            random_index1, random_index2 = np.random.choice(valid_indices, size=2, replace=False)   
            resample_factor =  data_pre_proc.shape[1] / data.shape[1] # since p_wave_index is updated with the new pre processed data
            random_index2 = int(random_index2 * resample_factor)

            unprocessed_noise_data = data[:, random_index1:random_index1 + window_size]
            noise_data = data_pre_proc[:, random_index2:random_index2 + window_size]
            
            # Ensure the extracted noise sample has the correct shape

            if noise_data.shape[1] < window_size:
                padding = np.random.normal(0, 0.00001, (data.shape[0], window_size - noise_data.shape[1]))
                noise_data = np.concatenate((noise_data, padding), axis=1)

            if p_data is not None and p_data.shape[1] < window_size:
                padding = np.random.normal(0, 0.000001, (p_data.shape[0], window_size - p_data.shape[1]))
                p_data = np.concatenate((padding, p_data), axis=1)

            if unprocessed_noise_data.shape[1] < window_size:
                padding = np.random.normal(0, 0.00001, (data.shape[0], window_size - unprocessed_noise_data.shape[1]))
                unprocessed_noise_data = np.concatenate((unprocessed_noise_data, padding), axis=1)

            if np.random.choice([True, False]):
                NOISE_MEAN = 0  # Gaussian noise mean
                NOISE_STD = 0.00001  # Gaussian noise std deviation
                unprocessed_noise_data += np.random.normal(NOISE_MEAN, NOISE_STD, unprocessed_noise_data.shape)

            if ( (len(p_data[0]) != window_size) or (len(noise_data[0]) != window_size) or (len(unprocessed_noise_data[0]) != window_size)):
                print("Wrong data  ====== : "+event_id)
                continue

            ## Add data to each groups
            if event_id not in positive_group_p:
                positive_p_dataset = positive_group_p.create_dataset(event_id, data=p_data)
            else:
                print(f"Dataset {event_id} already exists in positive_samples_p. Skipping.")

            if event_id not in negative_group:
                negative_dataset = negative_group.create_dataset(event_id, data=noise_data)
                negative_dataset = negative_group.create_dataset(event_id+"_stead", data=unprocessed_noise_data)
            else:
                print(f"Dataset {event_id} already exists in negative_group. Skipping.")

            count += 1

            for key, value in dataset.attrs.items():
                positive_group_p[event_id].attrs[key] = value

                negative_group[event_id].attrs[key] = value
                # Change the wave start time, samppling rate and other changed attributes

            print(f" {str(count)} : {event_id}")

    print ("Number of P records " + str(count))
    print(f"Number of records P arrival data is not available {non_p_arrival_count}")



## Main function to extract data from Geonet database and create 2s  dataset
def extract_data(cfg=None):

    print("Extracting data from the 90  geonet database")
    if cfg is None:
        cfg = Config()

    hdf5_file = h5py.File(cfg.ORIGINAL_DB_FILE, 'r')

    if os.path.isfile(cfg.DATA_EXTRACTED_FILE):
        os.remove(cfg.DATA_EXTRACTED_FILE)

    #stead_noise_data = np.load("stead_samples_80000.npy")

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
        stead_noise_index = 0

        #split_index = int(0.8 * len(hdf5_file.keys()))
        
        for event_id in hdf5_file.keys():  # Directly iterate over the keys in the HDF5 file

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

            # Only consider the reciords where epicentral distance is less than  100km
            if dataset.attrs["magnitude"] < 5 and  dataset.attrs["epicentral_distance"] > 100:
                continue

            if dataset.attrs["epicentral_distance"] > 150:
                continue

            data_pre_proc = pre_proc_data(data, sampling_rate=sampling_rate)

            # if sampling_rate != cfg.BASE_SAMPLING_RATE:
            #     data_pre_proc = downsample(data_pre_proc, cfg.BASE_SAMPLING_RATE, sampling_rate)
            #     downsample_factor = int(sampling_rate // cfg.BASE_SAMPLING_RATE)
            #     p_arrival_index = int(p_arrival_index/downsample_factor)
            #     s_arrival_index = int(s_arrival_index/downsample_factor)
            #     sampling_rate = cfg.BASE_SAMPLING_RATE

            if sampling_rate != cfg.BASE_SAMPLING_RATE:
                original_length = data_pre_proc.shape[1]
                data_pre_proc = resample_data(data_pre_proc, sampling_rate, cfg.BASE_SAMPLING_RATE)
                new_length = data_pre_proc.shape[1]
                # Adjust indices proportionally for both upsampling and downsampling
                p_arrival_index = int(p_arrival_index * new_length / original_length)
                s_arrival_index = int(s_arrival_index * new_length / original_length)
                sampling_rate = cfg.BASE_SAMPLING_RATE
                
            count += 1
            
            ## Give temporal shift to the data
            SHIFT_RANGE_SEC = 0.5  # Max shift in seconds (±0.5 sec)
            
            shift_samples = int(SHIFT_RANGE_SEC * sampling_rate)
            random_shifts_p = np.random.randint(-shift_samples, shift_samples + 1)
            random_shifts_s = np.random.randint(-shift_samples, shift_samples + 1)

            # Modify P and S indices
            new_p_indices = np.clip(p_arrival_index + random_shifts_p, 0, len(data[0]) - 1)
            new_s_indices = np.clip(s_arrival_index + random_shifts_s, 0, len(data[0]) - 1)

            # Extract wave data
            window_size = int(cfg.TRAINING_WINDOW * sampling_rate)
            p_data  = extract_wave_window(data_pre_proc, new_p_indices, window_size)
            s_data = extract_wave_window(data_pre_proc, new_s_indices, window_size)

            # Add noise samples
            start_index = 0
            end_index = max(1, p_arrival_index - window_size)  # Ensure at least one sample is available
            random_index1 = np.random.randint(start_index, end_index)
            random_index2 = np.random.randint(start_index, end_index)
            
            noise_data = data_pre_proc[:, random_index1:random_index1 + window_size]
            unprocessed_noise_data = data[:, random_index2:random_index2 + window_size]
            
            # Ensure the extracted noise sample has the correct shape

            if noise_data.shape[1] < window_size:
                padding = np.random.normal(0, 0.00001, (data.shape[0], window_size - noise_data.shape[1]))
                noise_data = np.concatenate((noise_data, padding), axis=1)

            if unprocessed_noise_data.shape[1] < window_size:
                padding = np.random.normal(0, 0.00001, (data.shape[0], window_size - unprocessed_noise_data.shape[1]))
                unprocessed_noise_data = np.concatenate((unprocessed_noise_data, padding), axis=1)

            if np.random.choice([True, False]):
                NOISE_MEAN = 0  # Gaussian noise mean
                NOISE_STD = 0.00001  # Gaussian noise std deviation
                unprocessed_noise_data += np.random.normal(NOISE_MEAN, NOISE_STD, unprocessed_noise_data.shape)

            if ( (len(p_data[0]) != window_size) or (len(s_data[0]) != window_size) or (len(noise_data[0]) != window_size) or (len(unprocessed_noise_data[0]) != window_size)):
                print("Wrong data  ====== : "+event_id)
                continue

            ## Add data to each groups
            if event_id not in positive_group_p:
                positive_p_dataset = positive_group_p.create_dataset(event_id, data=p_data)
            else:
                print(f"Dataset {event_id} already exists in positive_samples_p. Skipping.")

            if event_id not in positive_group_s:
                positive_s_dataset = positive_group_s.create_dataset(event_id, data=s_data)
            else:
                print(f"Dataset {event_id} already exists in positive_samples_p. Skipping.")

            if event_id not in negative_group:
                negative_dataset = negative_group.create_dataset(event_id, data=noise_data)
                negative_dataset = negative_group.create_dataset(event_id+"_stead", data=unprocessed_noise_data)
            else:
                print(f"Dataset {event_id} already exists in negative_group. Skipping.")

            for key, value in dataset.attrs.items():
                positive_group_p[event_id].attrs[key] = value
                #positive_group_s[event_id].attrs[key] = value
                negative_group[event_id].attrs[key] = value
                # Change the wave start time, samppling rate and other changed attributes

            print(f" {str(count)} : {event_id}")

    print ("Number of records " + str(count))
