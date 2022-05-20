import os
import pickle
import numpy as np


def load_dataset(dataset_name, condition_size, val_rate, test_rate, dataset_dir, horizon=1):
    dataset_cond, dataset_pred = (None,)*2

    if dataset_name == "lorenz":
        dataset_dir = os.path.join(dataset_dir, "lorenz")
        with open(dataset_dir + "/lorenz_dataset.pickle", "rb") as infile:
            dataset = pickle.load(infile)
        x_train = np.concatenate(list(dataset["x_train"].values()))
        y_train = np.concatenate(list(dataset["y_train"].values()))
        x_val = np.concatenate(list(dataset["x_val"].values()))
        y_val = np.concatenate(list(dataset["y_val"].values()))
        x_test = np.concatenate(list(dataset["x_test"].values()))
        y_test = np.concatenate(list(dataset["y_test"].values()))
        dataset_cond = {"train": x_train, "val": x_val, "test": x_test}
        dataset_pred = {"train": y_train, "val": y_val, "test": y_test}

    else:
        dataset_dir = os.path.join(dataset_dir, dataset_name)
        with open(dataset_dir + "/" + dataset_name + "_dataset.pickle", "rb") as infile:
            dataset = pickle.load(infile).astype(float)
        dataset_cond, dataset_pred = split_dataset(dataset, condition_size, horizon, val_rate, test_rate)

    return dataset_cond, dataset_pred


# Load lorenz test dataset considering the conditions
def load_dataset_per_condition(dataset_dir):
    dataset_dir = os.path.join(dataset_dir, "lorenz")
    with open(dataset_dir + "/lorenz_dataset.pickle", "rb") as infile:
        dataset = pickle.load(infile)
    x_test = dataset["x_test"]
    y_test = dataset["y_test"]
    return x_test, y_test


# Split dataset into training, validation and test
def split_dataset(dataset, condition_size, horizon=1, val_rate=0.1, test_rate=0.2):
    data_x = [dataset[i - condition_size:i] for i in range(condition_size, dataset.shape[0], horizon)]
    data_y = []
    for i in range(condition_size, dataset.shape[0], horizon):
        if (i+horizon) >= dataset.shape[0]:
            data_x = data_x[:-1]
        else:
            data_y.append(dataset[i:i + horizon])
    data_x = np.array(data_x)
    data_y = np.array(data_y)

    test_idx = int(data_x.shape[0] * (1 - test_rate))
    val_idx = int(data_x[:test_idx].shape[0] * (1 - val_rate))

    x_train, x_val, x_test = data_x[:val_idx], data_x[val_idx:test_idx], data_x[test_idx:]
    y_train, y_val, y_test = data_y[:val_idx], data_y[val_idx:test_idx], data_y[test_idx:]
    # y_train, y_val, y_test = y_train.reshape(-1, 1), y_val.reshape(-1, 1), y_test.reshape(-1, 1)

    dataset_cond = {"train": x_train, "val": x_val, "test": x_test}
    dataset_pred = {"train": y_train, "val": y_val, "test": y_test}

    return dataset_cond, dataset_pred
