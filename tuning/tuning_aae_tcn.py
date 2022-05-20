import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.optim as optim

import datasets
from models.aae import Encoder, Decoder, Discriminator


class Objective(object):
    """
    Tuning of the Adversarial auto-encoder developed on TCN cells
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
        tuning_params = {"ae_max_iter": trial.suggest_int("ae_max_iter", 1, 5),
                         "dis_max_iter": trial.suggest_int("dis_max_iter", 1, 5),
                         "enc_size": trial.suggest_categorical("enc_hidden_size", [8, 16, 32, 64, 128, 512]),
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

        encoder = Encoder(condition_size=tuning_params["condition_size"],
                          hidden_size=tuning_params["enc_size"],
                          latent_length=tuning_params["latent_length"],
                          cell_type="TCN",
                          num_channel=tuning_params["enc_num_channel"],
                          num_layers=tuning_params["num_layers"],
                          kernel_size=tuning_params["kernel_size"],
                          dropout_rate=tuning_params["enc_dropout_rate"],
                          mean=data_mean,
                          std=data_std).to(self.device)
        decoder = Decoder(condition_size=tuning_params["condition_size"],
                          hidden_size=tuning_params["dec_size"],
                          latent_length=tuning_params["latent_length"],
                          cell_type="TCN",
                          num_channel=tuning_params["dec_num_channel"],
                          num_layers=tuning_params["num_layers"],
                          kernel_size=tuning_params["kernel_size"],
                          dropout_rate=tuning_params["dec_dropout_rate"],
                          mean=data_mean,
                          std=data_std).to(self.device)
        discriminator = Discriminator(input_size=tuning_params["latent_length"]).to(self.device)
        optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop"])
        lr = trial.suggest_categorical("learning_rate", [1e-4, 1e-3, 1e-2])
        optimizer_encoder = getattr(optim, optimizer_name)(encoder.parameters(), lr=lr)
        optimizer_decoder = getattr(optim, optimizer_name)(decoder.parameters(), lr=lr)
        if optimizer_name == "Adam":
            optimizer_discriminator = getattr(optim, optimizer_name)(discriminator.parameters(), lr=lr,
                                                                     betas=(0.5, 0.999))
            optimizer_generator = getattr(optim, optimizer_name)(encoder.parameters(), lr=lr, betas=(0.5, 0.999))
        else:
            optimizer_discriminator = getattr(optim, optimizer_name)(discriminator.parameters(), lr=lr)
            optimizer_generator = getattr(optim, optimizer_name)(encoder.parameters(), lr=lr)
        criterion_BCE = nn.BCELoss().to(self.device)
        criterion_MSE = nn.MSELoss().to(self.device)

        crps, best_crps = (np.inf,) * 2

        # Train the model
        for step in range(1, self.max_steps + 1):
            try:
                # Autoencoder
                loss_reconst = 0
                for _ in range(tuning_params["ae_max_iter"]):
                    idx = rs.choice(x_train.shape[0], self.batch_size)
                    condition = x_train[idx]
                    real_data = y_train[idx]
                    optimizer_encoder.zero_grad()
                    optimizer_decoder.zero_grad()
                    z_sample = encoder(real_data, condition)
                    d_reconst = decoder(z_sample, condition).flatten()
                    loss_ae = criterion_MSE(d_reconst, real_data.flatten())
                    loss_ae.backward()
                    optimizer_encoder.step()
                    optimizer_decoder.step()
                    loss_reconst += loss_ae.detach().cpu().numpy()
                loss_reconst = loss_reconst / tuning_params["ae_max_iter"]

                # Discriminator
                loss_dis = 0
                for _ in range(tuning_params["dis_max_iter"]):
                    discriminator.zero_grad()
                    z_real = torch.tensor(rs.normal(0, 1, (self.batch_size, tuning_params["latent_length"])),
                                          device=self.device,
                                          dtype=torch.float32)
                    d_real = discriminator(z_real).flatten()
                    loss_real = criterion_BCE(d_real, torch.full_like(d_real, 1, device=self.device))
                    loss_real.backward()
                    loss_dis += loss_real.detach().cpu().numpy()
                    idx = rs.choice(x_train.shape[0], self.batch_size)
                    condition = x_train[idx]
                    real_data = y_train[idx]
                    z_sample = encoder(real_data, condition)
                    d_fake = discriminator(z_sample).flatten()
                    loss_fake = criterion_BCE(d_fake, torch.full_like(d_fake, 0, device=self.device))
                    loss_fake.backward()
                    optimizer_discriminator.step()
                    loss_dis += loss_fake.detach().cpu().numpy()
                loss_dis = loss_dis / (2 * tuning_params["dis_max_iter"])

                # Generator
                optimizer_generator.zero_grad()
                z_sample = encoder(real_data, condition)
                d_fake = discriminator(z_sample).flatten()
                loss_gen = -1 * criterion_BCE(d_fake, torch.full_like(d_fake, 0, device=self.device))
                loss_gen.backward()
                optimizer_generator.step()
                loss_gen = loss_gen.detach().cpu().numpy()

                # Validation of the model
                preds = []
                for _ in range(200):
                    noise = torch.tensor(rs.normal(0, 1, (x_val.size(0), tuning_params["latent_length"])),
                                         device=self.device,
                                         dtype=torch.float32)
                    with torch.no_grad():
                        pred = decoder(noise, x_val).detach().cpu().numpy().flatten()
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
                    print("-- [Trial {}]\t[Step {}/{}]\t[Reconst loss: {}]\t[D loss: {}]\t[G loss: {}]\t\t[crps latent: {}]".
                          format(trial.number, step, self.max_steps, loss_reconst, loss_dis, loss_gen, crps))

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
