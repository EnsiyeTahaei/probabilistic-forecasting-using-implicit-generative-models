import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.optim as optim

import datasets
from models.gan import Generator, Discriminator


class Objective(object):
    """
    Tuning of the Generative Adversarial Network developed on RNN cells
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
        tuning_params = {"noise_dim": trial.suggest_int("noise_dim", 8, 40),
                         "dis_max_iter": trial.suggest_int("dis_max_iter", 1, 5),
                         # "gen_hidden_size": trial.suggest_int("gen_hidden_size", 8, 512, log=True),
                         "gen_size": trial.suggest_categorical("gen_hidden_size", [8, 16, 32, 64, 128, 512]),
                         "gen_hidden_depth": trial.suggest_int("gen_hidden_depth", 1, 3),
                         "dis_size": trial.suggest_categorical("dis_hidden_size", [8, 16, 32, 64, 128, 512]),
                         "dis_hidden_depth": trial.suggest_int("dis_hidden_depth", 1, 3),
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
        x_val = torch.tensor(dataset_x["val"], device=self.device, dtype=torch.float32)
        y_val = dataset_y["val"].flatten()

        # Build model
        generator = Generator(noise_dim=tuning_params["noise_dim"],
                              condition_size=tuning_params["condition_size"],
                              gen_hidden_size=tuning_params["gen_size"],
                              hidden_layer_depth=tuning_params["gen_hidden_depth"],
                              cell_type=tuning_params["cell_type"],
                              mean=data_mean,
                              std=data_std).to(self.device)
        discriminator = Discriminator(condition_size=tuning_params["condition_size"],
                                      dis_hidden_size=tuning_params["dis_size"],
                                      hidden_layer_depth=tuning_params["dis_hidden_depth"],
                                      cell_type=tuning_params["cell_type"],
                                      mean=data_mean,
                                      std=data_std).to(self.device)
        optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop"])
        lr = trial.suggest_categorical("learning_rate", [1e-4, 1e-3, 1e-2])
        if optimizer_name == "Adam":
            optimizer_generator = getattr(optim, optimizer_name)(generator.parameters(), lr=lr, betas=(0.5, 0.999))
            optimizer_discriminator = getattr(optim, optimizer_name)(discriminator.parameters(), lr=lr,
                                                                     betas=(0.5, 0.999))
        else:
            optimizer_generator = getattr(optim, optimizer_name)(generator.parameters(), lr=lr)
            optimizer_discriminator = getattr(optim, optimizer_name)(discriminator.parameters(), lr=lr)
        criterion = nn.BCELoss().to(self.device)

        crps, best_crps = (np.inf,) * 2

        # Train model
        for step in range(1, self.max_steps + 1):
            try:
                generator.train()
                discriminator.train()
                loss_dis = 0
                for _ in range(tuning_params["dis_max_iter"]):
                    idx = rs.choice(x_train.shape[0], self.batch_size)
                    condition = x_train[idx]
                    real_data = y_train[idx]
                    discriminator.zero_grad()
                    d_real = discriminator(real_data, condition)
                    loss_real = criterion(d_real, torch.full_like(d_real, 1, device=self.device))
                    loss_real.backward()
                    loss_dis += loss_real.detach().cpu().numpy()
                    noise = torch.tensor(rs.normal(0, 1, (condition.size(0), tuning_params["noise_dim"])),
                                         device=self.device,
                                         dtype=torch.float32)
                    fake_data = generator(noise, condition).detach()
                    d_fake = discriminator(fake_data, condition)
                    loss_fake = criterion(d_fake, torch.full_like(d_fake, 0, device=self.device))
                    loss_fake.backward()
                    optimizer_discriminator.step()
                    loss_dis += loss_fake.detach().cpu().numpy()
                loss_dis = loss_dis / (2 * tuning_params["dis_max_iter"])

                generator.zero_grad()
                noise = torch.tensor(rs.normal(0, 1, (condition.size(0), tuning_params["noise_dim"])),
                                     device=self.device,
                                     dtype=torch.float32)
                fake_data = generator(noise, condition)
                d_fake = discriminator(fake_data, condition)
                loss_gen = -1 * criterion(d_fake, torch.full_like(d_fake, 0, device=self.device))
                loss_gen.backward()
                optimizer_generator.step()

                # Validate model
                generator.eval()
                preds = []
                for _ in range(200):
                    noise = torch.tensor(rs.normal(0, 1, (x_val.size(0), tuning_params["noise_dim"])),
                                         device=self.device,
                                         dtype=torch.float32)
                    with torch.no_grad():
                        pred = generator(noise, x_val).detach().cpu().numpy().flatten()
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
                    print("-- [Trial {}]\t[Step {}/{}]\t[D loss: {}]\t[G loss: {}]"
                          .format(trial.number, step, self.max_steps, loss_dis, loss_gen))

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
