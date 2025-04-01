
from utils import *
from database_op import *
from config import Config, MODE_TYPE, MODEL_TYPE
from dataprep import pre_proc_data


cfg = Config()
nncfg = NNCFG()

checkpoint = torch.load(cfg.MODEL_FILE_NAME, map_location=torch.device('cpu'))
model = PWaveCNN(model_id=nncfg.model_id, window_size=cfg.SAMPLE_WINDOW_SIZE, conv1_filters=16, conv2_filters=4, fc1_neurons=30, kernel_size1=nncfg.kernal_size1, kernel_size2=nncfg.kernal_size2)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# test_tensor = torch.rand(3, 100)
# test_tensor = torch.tensor(test_tensor, dtype=torch.float32).unsqueeze(0) 

# traced_model = torch.jit.trace(model, test_tensor)
# torch.jit.save(traced_model, cfg.MODEL_FILE_NAME+"_tcr.pt")


hdf5_file = h5py.File(cfg.TEST_DATA, 'r')
p_data, s_data, noise_data = getWaveData(cfg, hdf5_file)

p_data      = np.array(p_data)
s_data      = np.array(s_data)
noise_data  = np.array(noise_data)

p_data = pre_proc_data(p_data)
s_data = pre_proc_data(s_data)
noise_data = pre_proc_data(noise_data)

true_vrt    = np.array([1] * len(p_data) + [1] * len(s_data) +[0] * len(noise_data))

test_vtr    = np.concatenate((p_data, s_data, noise_data))

# Convert to tensor
test_tensor = torch.tensor(test_vtr, dtype=torch.float32)

sample = test_tensor[0].unsqueeze(0)
traced_model = torch.jit.trace(model, sample)
torch.jit.save(traced_model, cfg.MODEL_FILE_NAME+"_ts.pt")

