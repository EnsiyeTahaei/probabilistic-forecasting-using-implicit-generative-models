import numpy as np
import optuna
import torch
import torch.optim as optim

import datasets
from models.deepar import DeepARModel
from deepar import NormalNLLLoss
import math


class Objective(object):
    """
    Tuning of the DeepAR developed on RNN cells
    :param config: config information
    :param device: cpu/cuda to be used as a device
    """
    def __init__(self, config, device):
        self.max_steps = config['max_steps']
        self.batch_size = config['batch_size']
        self.dataset_name = config['dataset_name']
        self.dataset_dir = config['dataset_dir']
        self.val_rate = config['val_rate']
        self.test_rate = config['test_rate']
        self.seed = config['seed']
        self.tune_dir = config['tune_dir']
        self.device = device

    def __call__(self, trial: optuna.Trial):
        # Setup parameters
        rs = np.random.RandomState(self.seed)
        prune = False
        tuning_params = {"size": trial.suggest_categorical("hidden_size", [8, 16, 32, 64, 128, 512]),
                         "hidden_depth": trial.suggest_int("hidden_depth", 2, 3),
                         "cell_type": trial.suggest_categorical("cell_type", ["LSTM", "GRU"])}

        if self.dataset_name == "lorenz":
            tuning_params["condition_size"] = 24
        else:
            tuning_params["condition_size"] = trial.suggest_int("condition_size", 8, 48)

        # Load dataset
        dataset_x, dataset_y = datasets.load_dataset(dataset_name=self.dataset_name,
                                                     condition_size=tuning_params["condition_size"],
                                                     val_rate=self.val_rate,
                                                     test_rate=self.test_rate,
                                                     dataset_dir=self.dataset_dir)
        data_mean = dataset_x["train"].mean()
        data_std = dataset_x["train"].std()
        x_train = torch.tensor(dataset_x["train"], device=self.device, dtype=torch.float32)
        y_train = torch.tensor(dataset_y["train"], device=self.device, dtype=torch.float32)
        mean_val = dataset_x["val"].mean()
        std_val = dataset_x["val"].std()
        x_val = torch.tensor(dataset_x["val"], device=self.device, dtype=torch.float32)
        y_val = torch.tensor(dataset_y["val"], device=self.device, dtype=torch.float32)
        norm_y_val = (y_val - mean_val) / std_val

        # Build model
        deepAR = DeepARModel(condition_size=tuning_params["condition_size"],
                             hidden_size=tuning_params["size"],
                             hidden_layer_depth=tuning_params["hidden_depth"],
                             cell_type=tuning_params["cell_type"],
                             mean=data_mean,
                             std=data_std).to(self.device)
        optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop"])
        lr = trial.suggest_categorical("learning_rate", [1e-4, 1e-3, 1e-2])
        optimizer = getattr(optim, optimizer_name)(deepAR.parameters(), lr=lr)
        criterion = NormalNLLLoss()

        loss_val, best_loss_val = (np.inf,) * 2

        # Train model
        for step in range(1, self.max_steps + 1):
            try:
                deepAR.train()

                idx = rs.choice(x_train.shape[0], self.batch_size)
                condition = x_train[idx]
                real_data = y_train[idx]
                norm_real_data = (real_data - data_mean) / data_std

                deepAR.zero_grad()
                p_mean, p_sigma = deepAR(condition)
                if (any([math.isnan(m) for m in p_mean])) or (any([math.isnan(s) for s in p_sigma])):
                    prune = True
                    break
                loss = criterion(p_mean, p_sigma, norm_real_data)
                loss.backward()
                optimizer.step()

                # Validate model
                deepAR.eval()
                preds = []
                eps = 1e-16
                with torch.no_grad():
                    p_mean, p_sigma = deepAR(x_val)
                    if (any([math.isnan(m) for m in p_mean])) or (any([math.isnan(s) for s in p_sigma])):
                        prune = True
                        break
                    loss_val = criterion(p_mean, p_sigma, norm_y_val)

                if prune:
                    break

                if loss_val <= best_loss_val and loss_val != np.inf:
                    best_loss_val = loss_val
                    with open(self.tune_dir + "/tuning_history.txt", "a") as f:
                        f.write("-- Trial {} / Step {}: !!! New best model !!! loss_val {}\n"
                                .format(trial.number, step, best_loss_val))
                    print("-- Trial {} / Step {}: !!! New best model !!! loss_val {}"
                          .format(trial.number, step, best_loss_val))
                if step % 10 == 0:
                    print("-- [Trial {}]\t[Step {}/{}]\t[deepAR loss: {}]\t[val loss: {}]"
                          .format(trial.number, step, self.max_steps, loss, loss_val))

                trial.report(loss_val, step)

            except Exception as e:
                with open(self.tune_dir + "/error_history.txt", "a") as f:
                    f.write("-- trial {} : !!! ERROR !!! {}\n".format(trial.number, e.args))
                with open(self.tune_dir + "/error_history.txt", "a") as f:
                    f.write("-- trial {} : !!! ERROR !!! {}\n".format(trial.number, str(e.__class__.__name__)))
                print("Error!", e.args, "occurred.")
                optuna.exceptions.OptunaError()

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        if prune:
            with open(self.tune_dir + "/error_history.txt", "a") as f:
                f.write("-- trial {} : !!! Pruned manually !!!\n".format(trial.number))
            raise optuna.exceptions.TrialPruned()

        return loss_val


def tune(device, study_name, storage, config):
    torch.cuda.empty_cache()
    study = optuna.create_study(study_name=study_name,
                                direction="minimize",
                                load_if_exists=True,
                                storage=storage)

    study.optimize(Objective(config, device,),
                   n_trials=30,
                   gc_after_trial=True)
