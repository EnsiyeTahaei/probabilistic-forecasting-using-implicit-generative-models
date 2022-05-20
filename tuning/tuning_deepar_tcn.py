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
    Tuning of the DeepAR developed on TCN cells
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
                         "dropout_rate": trial.suggest_float("dropout_rate", 0., 0.5),
                         "num_channel": trial.suggest_int("num_channel", 5, 125)}

        if self.dataset_name == "lorenz":
            tuning_params["condition_size"] = 24
            tuning_params["num_layers"] = trial.suggest_categorical("num_layers", [1, 2])
            if tuning_params["num_layers"] == 1:
                tuning_params["kernel_size"] = 12
            elif tuning_params["num_layers"] == 2:
                tuning_params["kernel_size"] = 5
        else:
            tuning_params["num_layers"] = trial.suggest_int("num_layers", 1, 4)
            if tuning_params["num_layers"] == 1:
                tuning_params["kernel_size"] = trial.suggest_int("kernel_size", 5, 24)  # condition_size: 9 - 47
            elif tuning_params["num_layers"] == 2:
                tuning_params["kernel_size"] = trial.suggest_int("kernel_size", 2, 8)  # condition_size: 7 - 43
            elif tuning_params["num_layers"] == 3:
                tuning_params["kernel_size"] = trial.suggest_int("kernel_size", 2, 4)  # condition_size: 15 - 43
            elif tuning_params["num_layers"] == 4:
                tuning_params["kernel_size"] = 2  # condition_size: 31
            tuning_params["condition_size"] = 1 + 2 * (tuning_params["kernel_size"] - 1) * \
                                              (2 ** tuning_params["num_layers"] - 1)

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
        x_val = torch.tensor(dataset_x["val"], device=self.device, dtype=torch.float32)
        y_val = dataset_y["val"].flatten()

        # Build model
        deepAR = DeepARModel(condition_size=tuning_params["condition_size"],
                             hidden_size=tuning_params["size"],
                             cell_type="TCN",
                             num_channel=tuning_params["num_channel"],
                             num_layers=tuning_params["num_layers"],
                             kernel_size=tuning_params["kernel_size"],
                             dropout_rate=tuning_params["dropout_rate"],
                             mean=data_mean,
                             std=data_std).to(self.device)
        optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop"])
        lr = trial.suggest_categorical("learning_rate", [1e-4, 1e-3, 1e-2])
        optimizer = getattr(optim, optimizer_name)(deepAR.parameters(), lr=lr)
        criterion = NormalNLLLoss()

        crps, best_crps = (np.inf,) * 2

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
                for _ in range(200):
                    with torch.no_grad():
                        p_mean, p_sigma = deepAR(x_val)
                        if (any([math.isnan(m) for m in p_mean])) or (any([math.isnan(s) for s in p_sigma])):
                            prune = True
                            break
                        pred = []
                        for idx in range(x_val.size(0)):
                            sample = torch.normal(mean=p_mean[idx][0],
                                                  std=p_sigma[idx][0] + eps,
                                                  size=(1,),
                                                  device=self.device,
                                                  dtype=torch.float32)
                            sample = sample * data_std + data_mean
                            pred.append(sample.detach().cpu().numpy().flatten())
                        pred = np.vstack(pred).flatten()
                        preds.append(pred)

                if prune:
                    break

                preds = np.vstack(preds)
                crps = np.absolute(preds[:100] - y_val).mean() - 0.5 * np.absolute(preds[:100] - preds[100:]).mean()

                if crps <= best_crps and crps != np.inf:
                    best_crps = crps
                    with open(self.tune_dir + "/tuning_history.txt", "a") as f:
                        f.write("-- Trial {} / Step {}: !!! New best model !!! CRPS {}\n"
                                .format(trial.number, step, best_crps))
                    print("-- Trial {} / Step {}: !!! New best model !!! CRPS {}"
                          .format(trial.number, step, best_crps))
                if step % 10 == 0:
                    print("-- [Trial {}]\t[Step {}/{}]\t[deepAR loss: {}]\t[crps latent: {}]"
                          .format(trial.number, step, self.max_steps, loss, crps))

                trial.report(crps, step)

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

        return crps


def tune(device, study_name, storage, config):
    torch.cuda.empty_cache()
    study = optuna.create_study(study_name=study_name,
                                direction="minimize",
                                load_if_exists=True,
                                storage=storage)

    study.optimize(Objective(config, device,),
                   n_trials=30,
                   gc_after_trial=True)
