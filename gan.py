import os
import pickle
import statistics
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

import utils
from models.gan import Generator, Discriminator


class GAN(object):
    """
    Generative Adversarial Network model
    :param config: config information
    :param dataset_x: train dataset
    :param dataset_y: test dataset
    """
    def __init__(self, config, dataset_x, dataset_y):
        # Optimization parameters
        self.max_steps = config['max_steps']
        self.max_iter = config['dis_max_iter']
        self.patience = config['patience']
        self.batch_size = config['batch_size']
        self.condition_size = config['condition_size']
        self.horizon = config['horizon']
        self.optimizer_name = config['optimizer_name']
        self.lr = config['lr']
        self.quantile = config['quantile']
        self.sample_size = config['sample_size']
        # Network parameters
        self.cell_type = config['cell_type']
        self.noise_dim = config['noise_dim']
        self.gen_hidden_size = config['gen_hidden_size']
        self.gen_hidden_depth = config['gen_hidden_depth']
        self.gen_dropout_rate = config['gen_dropout_rate']
        self.gen_num_channel = config['gen_num_channel']
        self.dis_hidden_size = config['dis_hidden_size']
        self.dis_hidden_depth = config['dis_hidden_depth']
        self.dis_dropout_rate = config['dis_dropout_rate']
        self.dis_num_channel = config['dis_num_channel']
        self.num_layers = config['num_layers']
        self.kernel_size = config['kernel_size']
        # Dataset parameters
        self.dataset_dir = config['dataset_dir']
        self.dataset_name = config['dataset_name']
        self.val_rate = config['val_rate']
        self.test_rate = config['test_rate']
        self.dataset_x = dataset_x
        self.dataset_y = dataset_y
        self.data_mean = self.dataset_x["train"].mean()
        self.data_std = self.dataset_x["train"].std()
        # Evaluation parameters
        self.hist_bins = config['hist_bins']
        self.hist_min = config['hist_min']
        self.hist_max = config['hist_max']
        # Visualization parameters
        self.result_dir = config['result_dir']
        self.models_dir = os.path.join(self.result_dir, "models")
        self.histograms_dir = os.path.join(self.result_dir, "predictions_histograms")
        utils.check_dir(self.models_dir)
        utils.check_dir(self.histograms_dir)
        # Setup parameters
        self.rs = np.random.RandomState(config['seed'])
        self.device = config['device']
        self.max_device = config['max_device']
        self.seed = config['seed']
        self.log_interval = config['log_interval']
        self.save_interval = config['save_interval']
        # Training, evaluating and testing parameters
        self.training_info = {}
        self.testing_info = {}
        # Submodules
        self.generator = Generator(noise_dim=self.noise_dim,
                                   condition_size=self.condition_size,
                                   gen_hidden_size=self.gen_hidden_size,
                                   hidden_layer_depth=self.gen_hidden_depth,
                                   cell_type=self.cell_type,
                                   num_channel=self.gen_num_channel,
                                   num_layers=self.num_layers,
                                   kernel_size=self.kernel_size,
                                   dropout_rate=self.gen_dropout_rate,
                                   mean=self.data_mean,
                                   std=self.data_std).to(self.device)
        self.discriminator = Discriminator(condition_size=self.condition_size,
                                           dis_hidden_size=self.dis_hidden_size,
                                           hidden_layer_depth=self.dis_hidden_depth,
                                           cell_type=self.cell_type,
                                           num_channel=self.dis_num_channel,
                                           num_layers=self.num_layers,
                                           kernel_size=self.kernel_size,
                                           dropout_rate=self.dis_dropout_rate,
                                           mean=self.data_mean,
                                           std=self.data_std).to(self.device)
        if self.optimizer_name == "Adam":
            self.optimizer_generator = getattr(optim, self.optimizer_name)(self.generator.parameters(),
                                                                           lr=self.lr,
                                                                           betas=(0.5, 0.999))
            self.optimizer_discriminator = getattr(optim, self.optimizer_name)(self.discriminator.parameters(),
                                                                               lr=self.lr,
                                                                               betas=(0.5, 0.999))
        else:
            self.optimizer_generator = getattr(optim, self.optimizer_name)(self.generator.parameters(),
                                                                           lr=self.lr)
            self.optimizer_discriminator = getattr(optim, self.optimizer_name)(self.discriminator.parameters(),
                                                                               lr=self.lr)

        self.BCELoss = nn.BCELoss().to(self.device)
        # self.scheduler_generator = ReduceLROnPlateau(optimizer=self.optimizer_generator,
        #                                              mode='min',
        #                                              patience=20,
        #                                              verbose=True)
        # self.scheduler_discriminator = ReduceLROnPlateau(optimizer=self.optimizer_discriminator,
        #                                                  mode='min',
        #                                                  patience=20,
        #                                                  verbose=True)

        # Print networks architecture
        print("***********************************")
        print("*****  Networks architecture  *****")
        print(self.generator)
        print(self.discriminator)
        print("***********************************")

    def train(self):
        print("*********  Training model  ********")

        self.training_info['dis_loss'] = []
        self.training_info['gen_loss'] = []
        self.training_info['crps'] = []
        self.training_info['per_step_time'] = []
        self.training_info['total_time'] = []

        x_train = torch.tensor(self.dataset_x["train"], device=self.device, dtype=torch.float32)
        y_train = torch.tensor(self.dataset_y["train"], device=self.device, dtype=torch.float32)
        x_val = torch.tensor(self.dataset_x["val"], device=self.device, dtype=torch.float32)
        y_val = self.dataset_y["val"].flatten()
        best_crps = np.inf
        no_improve, stop = 0, False

        start_time = time.time()

        for step in range(1, self.max_steps + 1):
            step_start_time = time.time()
            loss_dis = 0
            for _ in range(self.max_iter):
                idx = self.rs.choice(x_train.shape[0], self.batch_size)
                condition = x_train[idx]
                real_data = y_train[idx]

                # Training discriminator on real data
                self.discriminator.zero_grad()
                d_real = self.discriminator(real_data, condition)
                loss_real = self.BCELoss(d_real, torch.full_like(d_real, 1, device=self.device))
                loss_real.backward()
                loss_dis += loss_real.detach().cpu().numpy()
                # Training discriminator on fake data
                noise = torch.tensor(self.rs.normal(0, 1, (condition.size(0), self.noise_dim)), device=self.device,
                                     dtype=torch.float32)
                fake_data = self.generator(noise, condition).detach()
                d_fake = self.discriminator(fake_data, condition)
                loss_fake = self.BCELoss(d_fake, torch.full_like(d_fake, 0, device=self.device))
                loss_fake.backward()
                self.optimizer_discriminator.step()
                loss_dis += loss_fake.detach().cpu().numpy()

            loss_dis = loss_dis / (2 * self.max_iter)

            # Training generator
            self.generator.zero_grad()
            noise = torch.tensor(self.rs.normal(0, 1, (self.batch_size, self.noise_dim)), device=self.device,
                                 dtype=torch.float32)
            fake_data = self.generator(noise, condition)
            d_fake = self.discriminator(fake_data, condition)
            # Datasets produce their best result with non-saturated loss
            loss_gen = -1 * self.BCELoss(d_fake, torch.full_like(d_fake, 0, device=self.device))
            loss_gen.backward()
            self.optimizer_generator.step()
            loss_gen = loss_gen.detach().cpu().numpy()

            # Validation
            preds = []
            for _ in range(200):
                noise = torch.tensor(self.rs.normal(0, 1, (x_val.size(0), self.noise_dim)), device=self.device,
                                     dtype=torch.float32)
                with torch.no_grad():
                    pred = self.generator(noise, x_val).detach().cpu().numpy().flatten()
                    preds.append(pred)

            preds = np.vstack(preds)
            crps = np.absolute(preds[:100] - y_val).mean() - 0.5 * np.absolute(preds[:100] - preds[100:]).mean()

            if crps <= best_crps and crps != np.inf:
                best_crps = crps
                print("-- Step {}: !!! New best model !!! CRPS {}".format(step, best_crps))
                torch.save({'gen_state_dict': self.generator.state_dict()}, self.result_dir + "/best_model")
                no_improve = 0
            else:
                no_improve += 1
                if no_improve == self.patience:
                    stop = True

            # Logging progress
            if step % self.log_interval == 0:
                print("-- [Step {}/{}]\t[D loss: {}]\t[G loss: {}]\t[crps latent: {}]".
                      format(step, self.max_steps, loss_dis, loss_gen, crps))

            self.save_model(step=step)
            self.save_info(info_label="training")
            step_end_time = time.time()
            step_time = step_end_time - step_start_time
            self.training_info['per_step_time'].append(step_time)
            self.training_info['dis_loss'].append(loss_dis)
            self.training_info['gen_loss'].append(loss_gen)
            self.training_info['crps'].append(crps)

            # self.scheduler_generator.step(crps)
            # self.scheduler_discriminator.step(crps)

            if stop:
                print("-- Early stopping on step {}".format(step))
                break

        end_time = time.time()
        total_time = end_time - start_time
        self.training_info['total_time'].append(total_time)
        print("-- Average time of one step: %.2f" % (np.mean(self.training_info['per_step_time'])))
        print("-- Total training time: %.2f" % (np.mean(self.training_info['total_time'][0])))
        self.save_info(info_label="training")
        print("*******  Training finished  *******")

    def test(self):
        print("*********  Testing model  *********")
        self.testing_info['preds'] = []
        self.testing_info['scores'] = []

        x_test = torch.tensor(self.dataset_x["test"], device=self.device, dtype=torch.float32)
        y_test = self.dataset_y["test"].flatten()

        self.load()

        for _ in range(200):
            noise = torch.tensor(self.rs.normal(0, 1, (x_test.size(0), self.noise_dim)),
                                 device=self.device,
                                 dtype=torch.float32)
            with torch.no_grad():
                pred = self.generator(noise, x_test).detach().cpu().numpy().flatten()
                self.testing_info['preds'].append(pred)

        self.testing_info['preds'] = np.vstack(self.testing_info['preds'])
        crps_score = np.absolute(self.testing_info['preds'][:100] - y_test).mean() - 0.5 * \
                     np.absolute(self.testing_info['preds'][:100] - self.testing_info['preds'][100:]).mean()
        self.testing_info['preds'] = self.testing_info['preds'].flatten()

        self.testing_info['scores'].append({"crps_score": crps_score})

        print("-- Test results:\t[CRPS {}]".format(crps_score))

        self.save_info(info_label="testing")
        title = "The predictions for the entire test dataset"
        utils.visualize_predictions(predictions=self.testing_info['preds'],
                                    ground_truth=y_test,
                                    title=title,
                                    bins=self.hist_bins,
                                    range_min=self.hist_min,
                                    range_max=self.hist_max,
                                    destination=self.histograms_dir)

    def test_on_lorenz(self, x_test, y_test):
        print("-- Testing model on whole conditions of lorenz  dataset")
        for i in range(5):
            x_test_cond = torch.tensor(x_test["condition_{}".format(i)], device=self.device, dtype=torch.float32)
            y_test_cond = y_test["condition_{}".format(i)].flatten()
            preds = self.test_per_condition(x_test=x_test_cond)
            title = "The predictions for condition " + str(i)
            utils.visualize_predictions(predictions=preds,
                                        ground_truth=y_test_cond,
                                        title=title,
                                        bins=self.hist_bins,
                                        range_min=self.hist_min,
                                        range_max=self.hist_max,
                                        destination=self.histograms_dir,
                                        condition=i)
            print("-- Condition " + str(i) + " finished.")

        print("-- Testing model on random conditions of lorenz  dataset")
        for i in range(5):
            x_test_cond = torch.tensor(x_test["condition_{}".format(i)], device=self.device, dtype=torch.float32)
            y_test_cond = y_test["condition_{}".format(i)].flatten()
            idx = self.rs.choice(x_test_cond.shape[0], 2)
            random_conditions = x_test_cond[idx]
            preds = self.test_per_condition(x_test=random_conditions)
            title = "The predictions for two randomly selected time window of condition " + str(i)
            utils.visualize_predictions(predictions=preds,
                                        ground_truth=y_test_cond,
                                        title=title,
                                        bins=self.hist_bins,
                                        range_min=self.hist_min,
                                        range_max=self.hist_max,
                                        destination=self.histograms_dir,
                                        condition=i)
            print("-- Condition " + str(i) + " finished.")

        print("-- Results saved.")

    def test_per_condition(self, x_test):
        preds = []
        self.load()
        for _ in range(200):
            noise = torch.tensor(self.rs.normal(0, 1, (x_test.size(0), self.noise_dim)),
                                 device=self.device,
                                 dtype=torch.float32)
            pred = self.generator(noise, x_test).detach().cpu().numpy().flatten()
            preds.append(pred)
        preds = np.vstack(preds)
        preds = preds.flatten()
        return preds

    def save_model(self, step):
        torch.save(self.generator.state_dict(),
                   self.models_dir + "/generator_parameters_" + str(step) + ".pth")
        torch.save(self.discriminator.state_dict(),
                   self.models_dir + "/discriminator_parameters_" + str(step) + ".pth")

    def save_info(self, info_label):
        if info_label == "training":
            with open(self.result_dir + '/training_info.pkl', 'wb') as f:
                pickle.dump(self.training_info, f)
        elif info_label == "testing":
            with open(self.result_dir + '/testing_info.pkl', 'wb') as f:
                pickle.dump(self.testing_info, f)

    def load(self):
        checkpoint = torch.load(self.result_dir + "/best_model", map_location=torch.device('cpu'))
        self.generator.load_state_dict(checkpoint['gen_state_dict'])

    def forecast(self):
        forecast_dir = os.path.join(self.result_dir, "forecasting")
        forecast_dir = os.path.join(forecast_dir, str(self.quantile))
        utils.check_dir(forecast_dir)

        x_test = torch.tensor(self.dataset_x["test"], device=self.device, dtype=torch.float32)
        y_test = torch.tensor(self.dataset_y["test"], dtype=torch.float32)
        self.load()
        total_preds = []
        crps_scores = []

        for i, cond in enumerate(x_test):
            noise = torch.tensor(self.rs.normal(0, 1, (self.sample_size, self.noise_dim)),
                                 device=self.device,
                                 dtype=torch.float32)
            cond = cond.view(1, -1).repeat(self.sample_size, 1)
            with torch.no_grad():
                pred = self.generator(noise, cond).detach().cpu()
            crps = np.absolute(pred[:100] - y_test[i][0]).mean() - 0.5 * np.absolute(pred[:100] - pred[100:]).mean()
            crps_scores.append(crps.cpu().numpy().flatten()[0])
            preds = pred.view(-1, 1).to(self.device)
            for h in range(self.horizon-1):
                sorted_pred, _ = pred.sort()
                medians = torch.empty((self.sample_size, 1), device=self.device, dtype=torch.float32)
                q_factors, q_sizes = utils.load_quantiles(quantile=self.quantile, sample_size=self.sample_size)
                start_point = 0
                for q_idx, q in enumerate(q_factors):
                    median = sorted_pred[int(self.sample_size * q)]
                    median = median.view(1, -1).repeat(q_sizes[q_idx], 1)
                    medians[start_point:start_point + q_sizes[q_idx]] = median
                    start_point = start_point + int(q_sizes[q_idx])
                cond = torch.cat((cond, medians), dim=1)[:, 1:]
                noise = torch.tensor(self.rs.normal(0, 1, (self.sample_size, self.noise_dim)),
                                     device=self.device,
                                     dtype=torch.float32)
                with torch.no_grad():
                    pred = self.generator(noise, cond).detach().cpu()
                crps = np.absolute(pred[:100] - y_test[i][h+1]).mean() - 0.5 * np.absolute(pred[:100] - pred[100:]).mean()
                crps_scores.append(crps.cpu().numpy().flatten()[0])
                preds = torch.cat((preds, pred.view(-1, 1).to(self.device)), dim=1)
            title = "The predictions for condition window " + str(i)
            total_preds.append(preds.cpu())
            utils.visualize_crps(predictions=preds.cpu(),
                                 ground_truth=y_test[i],
                                 cond_idx=i,
                                 horizon=self.horizon,
                                 title=title,
                                 destination=forecast_dir)
        final_crps = statistics.mean(crps_scores)
        print("-- Final CRPS = ", final_crps)
        with open(forecast_dir + '/crps_info.pkl', 'wb') as f:
            pickle.dump(crps_scores, f)
        with open(forecast_dir + '/preds_info.pkl', 'wb') as f:
            pickle.dump(total_preds, f)
