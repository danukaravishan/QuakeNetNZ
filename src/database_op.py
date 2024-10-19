from utils import *
from config import *

def resample_data(data, frequency):
    resampled_data = data.resample(frequency).mean()
    return resampled_data

def extract_p_wave_window(data, p_wave_index, window_size):
    end = p_wave_index + window_size
    return data[p_wave_index:end]

def extract_noise_window(data, window_size):
    return data[-window_size:]


def extractDataFromHDF5Group(group):
    data_list = []
    for event_id in group.keys():
        dataset = group[event_id]
        data = np.array(dataset)  # Convert dataset to NumPy array
        data_list.append(data)    # Append to list
    return data_list


def getWaveData(cfg, hdf5_file):

    positive_group_p = hdf5_file['positive_samples_p']
    positive_group_s = hdf5_file['positive_samples_s']
    negative_group   = hdf5_file['negative_sample_group']

    p_data = extractDataFromHDF5Group(positive_group_p)
    s_data = extractDataFromHDF5Group(positive_group_s)
    noise_data= extractDataFromHDF5Group(negative_group)

    hdf5_file.close()

    return p_data, s_data, noise_data

# This function will split the cfg.DATABASE_FILE into two (Train and Test) and create new files

def split_data():
    cfg = Config()
    hdf5_file = h5py.File(cfg.DATABASE_FILE, 'r')

    if not os.path.isfile(cfg.DATABASE_FILE):
        return Exception("Database file is not available")
    
    # Train dataset
    train_dataset_file = cfg.TRAIN_DATA
    # Test dataset
    test_dataset_file = cfg.TEST_DATA

    if os.path.isfile(train_dataset_file):
        os.remove(train_dataset_file)
    if os.path.isfile(test_dataset_file):
        os.remove(test_dataset_file)

    with h5py.File(cfg.DATABASE_FILE, 'r') as hdf_file:
        
        # Create new HDF5 files for train and test
        with h5py.File(train_dataset_file, 'w') as train_file, h5py.File(test_dataset_file, 'w') as test_file:
            
            # Loop through each group in the original HDF5 file (p_wave, s_wave, noise)
            for group_name in hdf_file.keys():

                if group_name not in train_file:
                    train_group = train_file.create_group(group_name)
                else:
                    train_group = train_file[group_name]
                    
                if group_name not in test_file:
                    test_group = test_file.create_group(group_name)
                else:
                    test_group = test_file[group_name]

                group = hdf_file[group_name]  # Load the entire dataset for this group
                
                count = 0
                split_index = int(0.8 * len(group))

                for event_id in group.keys():
                    data = group.get(event_id)
                    if (count < split_index):
                        if event_id not in train_group:
                            new_dataset = train_group.create_dataset(event_id, data=data)
                            for attr_name, attr_value in data.attrs.items():
                                new_dataset.attrs[attr_name] = attr_value
                        else:
                            print(f"Dataset {event_id} already exists in positive_samples_p. Skipping.")
                    else: 
                        if event_id not in test_group:
                            new_dataset = test_group.create_dataset(event_id, data=data)
                            for attr_name, attr_value in data.attrs.items():
                                new_dataset.attrs[attr_name] = attr_value
                        else:
                            print(f"Dataset {event_id} already exists in positive_samples_p. Skipping.")
                    
                    count +=1
                    print(event_id)