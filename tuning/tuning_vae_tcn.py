import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.optim as optim

import datasets
from models.vae import VAEModel
import math


class Objective(object):
    """
    Tuning of the Variational auto-encoder developed on TCN cells
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
        tuning_params = {"enc_size": trial.suggest_categorical("enc_hidden_size", [8, 16, 32, 64, 128, 512]),
                         "dec_size": trial.suggest_categorical("dec_hidden_size", [8, 16, 32, 64, 128, 512]),
                         "latent_length": trial.suggest_int("latent_length", 8, 48),
                         "enc_dropout_rate": trial.suggest_float("enc_dropout_rate", 0., 0.5),
                         "dec_dropout_rate": trial.suggest_float("dec_dropout_rate", 0., 0.5),
                         "enc_num_channel": trial.suggest_int("enc_num_channel", 5, 125),
                         "dec_num_channel": trial.suggest_int("dec_num_channel", 5, 125)}

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
        model = VAEModel(condition_size=tuning_params["condition_size"],
                         enc_hidden_size=tuning_params["enc_size"],
                         dec_hidden_size=tuning_params["dec_size"],
                         latent_length=tuning_params["latent_length"],
                         cell_type="TCN",
                         enc_dropout_rate=tuning_params["enc_dropout_rate"],
                         dec_dropout_rate=tuning_params["dec_dropout_rate"],
                         enc_num_channel=tuning_params["enc_num_channel"],
                         dec_num_channel=tuning_params["dec_num_channel"],
                         num_layers=tuning_params["num_layers"],
                         kernel_size=tuning_params["kernel_size"],
                         device=self.device,
                         mean=data_mean,
                         std=data_std).to(self.device)
        optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop"])
        lr = trial.suggest_categorical("learning_rate", [1e-4, 1e-3, 1e-2])
        optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
        criterion = nn.MSELoss().to(self.device)

        crps, best_crps = (np.inf,) * 2

        # Train model
        for step in range(1, self.max_steps + 1):
            try:
                model.train()
                idx = rs.choice(x_train.shape[0], self.batch_size)
                condition = x_train[idx]
                real_data = y_train[idx]
                model.zero_grad()
                d_reconst = model(real_data, condition)
                if any([math.isnan(item) for item in d_reconst]):
                    prune = True
                    break
                latent_mean, latent_logvar = model.reparameterize.latent_mean, model.reparameterize.latent_logvar
                kl_loss = -0.5 * torch.mean(1 + latent_logvar - latent_mean.pow(2) - latent_logvar.exp())
                reconst_loss = criterion(d_reconst.flatten(), real_data.flatten())
                loss = kl_loss + reconst_loss
                loss.backward()
                optimizer.step()

                # Validate model
                model.eval()
                preds = []
                for _ in range(200):
                    noise = torch.tensor(rs.normal(0, 1, (x_val.size(0), tuning_params["latent_length"])),
                                         device=self.device,
                                         dtype=torch.float32)
                    with torch.no_grad():
                        pred = model.decoder(noise, x_val).detach().cpu().numpy().flatten()
                        preds.append(pred)

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
                    print("-- [Trial {}]\t[Step {}/{}]\t[Reconst loss: {}]\t[KLD loss: {}]\t[crps latent: {}]".
                          format(trial.number, step, self.max_steps, reconst_loss, kl_loss, crps))

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

    study.optimize(Objective(config, device, ),
                   n_trials=30,
                   gc_after_trial=True)
