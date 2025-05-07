from utils import *
from database_op import *
from config import Config, MODE_TYPE, MODEL_TYPE, NNCFG
import optuna
from torch.utils.data import random_split, DataLoader, TensorDataset
from dataprep import pre_proc_data,normalize, normalize_data
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from report  import plot_loss
from train import _train


def data_loader():
    cfg = Config()
    nncfg = NNCFG()
    nncfg.argParser()

    hdf5_file = h5py.File(cfg.TRAIN_DATA, 'r')

    p_data, s_data, noise_data = getWaveData(cfg, hdf5_file)
        
    # Data preparation
    p_data = np.array(p_data)
    s_data = np.array(s_data)
    noise_data = np.array(noise_data)

    p_data = normalize_data(p_data)
    s_data = normalize_data(s_data)
    noise_data = normalize_data(noise_data)

    ## Merge First 20% of test data into validation set
    hdf5_file_test = h5py.File(cfg.TEST_DATA, 'r')
    p_data_test, s_data_test, noise_data_test = getWaveData(cfg, hdf5_file_test)

    p_data_test = np.array(p_data_test)
    s_data_test = np.array(s_data_test)
    noise_data_test = np.array(noise_data_test)

    test_val_split_ratio = 0.5
    
    random_state = np.random.RandomState(42)  # Set a fixed random seed for consistency
    random_indices = random_state.choice(len(p_data_test), int(test_val_split_ratio * p_data_test.shape[0]), replace=False)
    p_data_test = p_data_test[random_indices]

    random_indices = random_state.choice(len(s_data_test), int(test_val_split_ratio * s_data_test.shape[0]), replace=False)
    s_data_test = s_data_test[random_indices]

    random_indices = random_state.choice(len(noise_data_test), int(test_val_split_ratio * noise_data_test.shape[0]), replace=False)
    noise_data_test = noise_data_test[random_indices]

    p_data_test = normalize_data(p_data_test)
    s_data_test = normalize_data(s_data_test)
    noise_data_test = normalize_data(noise_data_test)
    
    ### 
    positive_data = np.concatenate((p_data , s_data))
    X = np.concatenate([positive_data, noise_data], axis=0)
    Y = np.array([1] * len(positive_data) + [0] * len(noise_data))  # 1 for P wave, 0 for noise

    full_dataset = list(zip(X, Y))
    random.Random(42).shuffle(full_dataset)
    X, Y = zip(*full_dataset)
    X = np.array(X)
    Y = np.array(Y)

    train_size = int(1 * len(X))

    X_train = X[:train_size]
    Y_train = Y[:train_size]

    X_val = X[train_size:]
    Y_val = Y[train_size:]

    X_test_val = np.concatenate([p_data_test, s_data_test, noise_data_test])
    Y_test_val = np.array([1] * (len(p_data_test) + len(s_data_test)) + [0] * len(noise_data_test))

    X_val = np.concatenate([X_val, X_test_val])
    Y_val = np.concatenate([Y_val, Y_test_val])
    return X_train, Y_train, X_val, Y_val, X_test_val, Y_test_val


def objective(trial, X_train, Y_train, X_val, Y_val):
    cfg = Config()
    nncfg = NNCFG()
    nncfg.argParser()


    conv1_filters_opt=32
    conv2_filters_opt=32 
    conv3_filters_opt=32 
    fc1_neurons_opt=44
    fc2_neurons_opt=18 
    kernel_size1_opt=4 
    kernel_size2_opt=4 
    kernel_size3_opt=4 
    dropout1_opt=0.3
    dropout2_opt=0.2 
    dropout3_opt=0.1
    l2_decay_opt=0.0008
    batch_size_opt=1024
    learning_rate_opt=0.002

    # Suggest hyperparameters
    conv1_filters_opt = trial.suggest_int("conv1_filters", 8, 32, step=4)
    conv2_filters_opt = trial.suggest_int("conv2_filters", 4, 32, step=4)
    conv3_filters_opt = trial.suggest_int("conv3_filters", 4, 32, step=4)
    fc1_neurons_opt = trial.suggest_int("fc1_neurons", 8, 64, step=4)
    fc2_neurons_opt = trial.suggest_int("fc2_neurons", 8, 64, step=4)
    dropout1_opt = trial.suggest_uniform("dropout1", 0.1, 0.5)
    dropout2_opt = trial.suggest_uniform("dropout2", 0.1, 0.5)
    dropout3_opt = trial.suggest_uniform("dropout3", 0.1, 0.5)
    l2_decay_opt = trial.suggest_loguniform("l2_decay", 1e-5, 1e-2)

    kernel_size1_opt = trial.suggest_int("kernel_size1", 2, 10, step=1)
    kernel_size2_opt = trial.suggest_int("kernel_size2", 2, 10, step=1)
    kernel_size3_opt = trial.suggest_int("kernel_size3", 2, 10, step=1)
    learning_rate_opt = trial.suggest_loguniform("learning_rate", 1e-4, 1e-2)
    batch_size_opt = trial.suggest_categorical("batch_size", [32, 256, 512, 1024, 2048])


    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                    torch.tensor(Y_train, dtype=torch.float)),
        batch_size=batch_size_opt,
        shuffle=True
    )
   
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                        torch.tensor(Y_val, dtype=torch.float)),
        batch_size=batch_size_opt,
        shuffle=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = None

    ## Train the model. For now, thinking that all the type of models can take same kind of input
    model = PWaveCNN( 
            window_size=cfg.SAMPLE_WINDOW_SIZE, 
            conv1_filters=conv1_filters_opt, 
            conv2_filters=conv2_filters_opt, 
            conv3_filters=conv3_filters_opt, 
            fc1_neurons=fc1_neurons_opt, 
            fc2_neurons=fc2_neurons_opt, 
            kernel_size1=kernel_size1_opt, 
            kernel_size2=kernel_size2_opt, 
            kernel_size3=kernel_size3_opt, 
            dropout1=dropout1_opt, 
            dropout2=dropout2_opt, 
            dropout3=dropout3_opt
    ).to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate_opt, weight_decay=l2_decay_opt)

    model, train_losses, val_loss, val_acc = _train(model, train_loader, val_loader, optimizer, criterion, 100)

    last_train_loss = float(train_losses[-1])
    last_val_loss = float(val_loss[-1])
    loss_gap = float(abs(last_val_loss - last_train_loss))
    val_loss_diff = np.diff(val_loss)
    val_loss_smoothness = float(np.std(val_loss_diff))

    weights = [1, 1, 2, 10]

    combined_score = (
        weights[0] * last_train_loss +
        weights[1] * last_val_loss +
        weights[2] * loss_gap +
        weights[3] * val_loss_smoothness
    )
    return combined_score


def hyper_param_opt():
    X_train, Y_train, X_val, Y_val, X_test_val, Y_test_val = data_loader()
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, X_train, Y_train, X_val, Y_val), n_trials=500)
    print("\n\nCompleted trials all the trials\n\n")
    print("Best trial(s):")
    with open("hyper_param.txt", "a") as f:
        f.write(str(study.best_trial.params) + "\n")

if __name__ == "__main__":
    hyper_param_opt()