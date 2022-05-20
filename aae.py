import os
import pickle
import statistics
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.spatial.distance import jensenshannon
from torch.optim.lr_scheduler import ReduceLROnPlateau

import utils
from models.aae import Encoder, Decoder, Discriminator


class AAE(object):
    """
    Adversarial auto-encoder model
    :param config: config information
    :param dataset_x: train dataset
    :param dataset_y: test dataset
    """
    def __init__(self, config, dataset_x, dataset_y):
        # Optimization parameters
        self.max_steps = config['max_steps']
        self.dis_max_iter = config['dis_max_iter']
        self.ae_max_iter = config['ae_max_iter']
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
        self.latent_length = config['latent_length']
        self.enc_hidden_size = config['enc_hidden_size']
        self.dec_hidden_size = config['dec_hidden_size']
        self.enc_hidden_depth = config['enc_hidden_depth']
        self.dec_hidden_depth = config['dec_hidden_depth']
        self.enc_dropout_rate = config['enc_dropout_rate']
        self.dec_dropout_rate = config['dec_dropout_rate']
        self.enc_num_channel = config['enc_num_channel']
        self.dec_num_channel = config['dec_num_channel']
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
        self.encoder = Encoder(condition_size=self.condition_size,
                               hidden_size=self.enc_hidden_size,
                               hidden_layer_depth=self.enc_hidden_depth,
                               latent_length=self.latent_length,
                               cell_type=self.cell_type,
                               num_channel=self.enc_num_channel,
                               num_layers=self.num_layers,
                               kernel_size=self.kernel_size,
                               dropout_rate=self.enc_dropout_rate,
                               mean=self.data_mean,
                               std=self.data_std).to(self.device)
        self.decoder = Decoder(condition_size=self.condition_size,
                               hidden_size=self.dec_hidden_size,
                               hidden_layer_depth=self.dec_hidden_depth,
                               latent_length=self.latent_length,
                               cell_type=self.cell_type,
                               num_channel=self.dec_num_channel,
                               num_layers=self.num_layers,
                               kernel_size=self.kernel_size,
                               dropout_rate=self.dec_dropout_rate,
                               mean=self.data_mean,
                               std=self.data_std).to(self.device)
        self.discriminator = Discriminator(input_size=self.latent_length).to(self.device)
        self.optimizer_encoder = getattr(optim, self.optimizer_name)(self.encoder.parameters(), lr=self.lr)
        self.optimizer_decoder = getattr(optim, self.optimizer_name)(self.decoder.parameters(), lr=self.lr)
        if self.optimizer_name == "Adam":
            self.optimizer_discriminator = getattr(optim, self.optimizer_name)(self.discriminator.parameters(),
                                                                               lr=self.lr, betas=(0.5, 0.999))
            self.optimizer_generator = getattr(optim, self.optimizer_name)(self.encoder.parameters(),
                                                                           lr=self.lr, betas=(0.5, 0.999))
        else:
            self.optimizer_discriminator = getattr(optim, self.optimizer_name)(self.discriminator.parameters(),
                                                                               lr=self.lr)
            self.optimizer_generator = getattr(optim, self.optimizer_name)(self.encoder.parameters(), lr=self.lr)
        self.BCELoss = nn.BCELoss().to(self.device)
        self.MSELoss = nn.MSELoss().to(self.device)
        # self.scheduler_encoder = ReduceLROnPlateau(optimizer=self.optimizer_encoder,
        #                                            mode='min',
        #                                            patience=20,
        #                                            verbose=True)
        # self.scheduler_decoder = ReduceLROnPlateau(optimizer=self.optimizer_decoder,
        #                                            mode='min',
        #                                            patience=20,
        #                                            verbose=True)
        # self.scheduler_discriminator = ReduceLROnPlateau(optimizer=self.optimizer_discriminator,
        #                                                  mode='min',
        #                                                  patience=20,
        #                                                  verbose=True)
        # self.scheduler_generator = ReduceLROnPlateau(optimizer=self.optimizer_generator,
        #                                              mode='min',
        #                                              patience=20,
        #                                              verbose=True)
        
        # Print networks architecture
        print("***********************************")
        print("*****  Networks architecture  *****")
        print(self.encoder)
        print(self.decoder)
        print(self.discriminator)
        print("***********************************")

    def train(self):
        print("*********  Training model  ********")

        self.training_info['dis_loss'] = []
        self.training_info['gen_loss'] = []
        self.training_info['jsd'] = []
        self.training_info['crps'] = []
        self.training_info['per_step_time'] = []
        self.training_info['total_time'] = []

        x_train = torch.tensor(self.dataset_x["train"], device=self.device, dtype=torch.float32)
        y_train = torch.tensor(self.dataset_y["train"], device=self.device, dtype=torch.float32)
        x_val = torch.tensor(self.dataset_x["val"], device=self.device, dtype=torch.float32)
        y_val = self.dataset_y["val"].flatten()
        best_jsd_latent = np.inf
        best_crps = np.inf
        no_improve, stop = 0, False

        start_time = time.time()

        for step in range(1, self.max_steps + 1):
            step_start_time = time.time()

            # Training autoencoder
            loss_reconst = 0
            for _ in range(self.ae_max_iter):
                idx = self.rs.choice(x_train.shape[0], self.batch_size)
                condition = x_train[idx]
                real_data = y_train[idx]

                self.encoder.zero_grad()
                self.decoder.zero_grad()
                z_sample = self.encoder(real_data, condition)
                d_reconst = self.decoder(z_sample, condition).flatten()
                loss_ae = self.MSELoss(d_reconst, real_data.flatten())
                loss_ae.backward()
                self.optimizer_encoder.step()
                self.optimizer_decoder.step()
                loss_reconst += loss_ae.detach().cpu().numpy()

            loss_reconst = loss_reconst / self.ae_max_iter

            # Training discriminator
            loss_dis = 0
            for _ in range(self.dis_max_iter):
                self.discriminator.zero_grad()
                # Real data
                z_real = torch.tensor(self.rs.normal(0, 1, (self.batch_size, self.latent_length)),
                                      device=self.device,
                                      dtype=torch.float32)
                d_real = self.discriminator(z_real).flatten()
                loss_real = self.BCELoss(d_real, torch.full_like(d_real, 1, device=self.device))
                loss_real.backward()
                loss_dis += loss_real.detach().cpu().numpy()
                # Fake data
                idx = self.rs.choice(x_train.shape[0], self.batch_size)
                condition = x_train[idx]
                real_data = y_train[idx]
                z_sample = self.encoder(real_data, condition)
                d_fake = self.discriminator(z_sample).flatten()
                loss_fake = self.BCELoss(d_fake, torch.full_like(d_fake, 0, device=self.device))
                loss_fake.backward()
                self.optimizer_discriminator.step()
                loss_dis += loss_fake.detach().cpu().numpy()
            loss_dis = loss_dis / (2 * self.dis_max_iter)

            # Training encoder as generator
            self.optimizer_generator.zero_grad()
            z_sample = self.encoder(real_data, condition)
            d_fake = self.discriminator(z_sample).flatten()
            loss_gen = -1 * self.BCELoss(d_fake, torch.full_like(d_fake, 0, device=self.device))
            loss_gen.backward()
            self.optimizer_generator.step()
            loss_gen = loss_gen.detach().cpu().numpy()

            # Test latent space
            jsd_latent = jensenshannon(
                np.histogram(z_real.cpu().numpy(), bins=100, range=(-11, 11), density=True)[0],
                np.histogram(z_sample.cpu().detach().numpy(), bins=100, range=(-11, 11), density=True)[0]
            )
            if jsd_latent <= best_jsd_latent and jsd_latent != np.inf:
                best_jsd_latent = jsd_latent
                print("-- Step {}: *** New best latent *** JSD {}".format(step, best_jsd_latent))
                with open(self.result_dir + "/latent_info.pkl", 'wb') as f:
                    pickle.dump(z_sample.cpu(), f)

            # Validation
            preds = []
            for _ in range(200):
                noise = torch.tensor(self.rs.normal(0, 1, (x_val.size(0), self.latent_length)),
                                     device=self.device,
                                     dtype=torch.float32)
                with torch.no_grad():
                    pred = self.decoder(noise, x_val).detach().cpu().numpy().flatten()
                    preds.append(pred)

            preds = np.vstack(preds)
            crps = np.absolute(preds[:100] - y_val).mean() - 0.5 * np.absolute(preds[:100] - preds[100:]).mean()

            if 600 < step and crps <= best_crps and crps != np.inf:
                best_crps = crps
                print("-- Step {}: !!! New best model !!! CRPS {}".format(step, best_crps))
                torch.save({'dec_state_dict': self.decoder.state_dict()}, self.result_dir + "/best_model")
                no_improve = 0
            else:
                no_improve += 1
                if no_improve == self.patience:
                    stop = True

            # Logging progress
            if step % self.log_interval == 0:
                print("-- [Step {}/{}]\t[Reconst loss: {}]\t[D loss: {}]\t[G loss: {}]\t[jsd latent: {}]"
                      "\t[crps latent: {}]".
                      format(step, self.max_steps, loss_reconst, loss_dis, loss_gen, jsd_latent, crps))

            self.save_model(step=step)
            self.save_info(info_label="training")
            step_end_time = time.time()
            step_time = step_end_time - step_start_time
            self.training_info['per_step_time'].append(step_time)
            self.training_info['dis_loss'].append(loss_dis)
            self.training_info['gen_loss'].append(loss_gen)
            self.training_info['jsd'].append(jsd_latent)
            self.training_info['crps'].append(crps)

            # self.scheduler_encoder.step(crps)
            # self.scheduler_decoder.step(crps)
            # self.scheduler_discriminator.step(crps)
            # self.scheduler_generator.step(crps)

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
            noise = torch.tensor(self.rs.normal(0, 1, (x_test.size(0), self.latent_length)),
                                 device=self.device,
                                 dtype=torch.float32)
            with torch.no_grad():
                pred = self.decoder(noise, x_test).detach().cpu().numpy().flatten()
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
            noise = torch.tensor(self.rs.normal(0, 1, (x_test.size(0), self.latent_length)),
                                 device=self.device,
                                 dtype=torch.float32)
            with torch.no_grad():
                pred = self.decoder(noise, x_test).detach().cpu().numpy().flatten()
                preds.append(pred)
        preds = np.vstack(preds)
        preds = preds.flatten()
        return preds

    def save_model(self, step):
        torch.save(self.encoder.state_dict(),
                   self.models_dir + "/encoder_parameters_" + str(step) + ".pth")
        torch.save(self.decoder.state_dict(),
                   self.models_dir + "/decoder_parameters_" + str(step) + ".pth")
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
        checkpoint = torch.load(self.result_dir + "/best_model")
        self.decoder.load_state_dict(checkpoint['dec_state_dict'])

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
            noise = torch.tensor(self.rs.normal(0, 1, (self.sample_size, self.latent_length)),
                                 device=self.device,
                                 dtype=torch.float32)
            cond = cond.view(1, -1).repeat(self.sample_size, 1)
            with torch.no_grad():
                pred = self.decoder(noise, cond).detach().cpu()
            crps = np.absolute(pred[:100] - y_test[i][0]).mean() - 0.5 * np.absolute(pred[:100] - pred[100:]).mean()
            crps_scores.append(crps.cpu().numpy().flatten()[0])
            preds = pred.view(-1, 1).to(self.device)
            for h in range(self.horizon - 1):
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
                noise = torch.tensor(self.rs.normal(0, 1, (self.sample_size, self.latent_length)),
                                     device=self.device,
                                     dtype=torch.float32)
                with torch.no_grad():
                    pred = self.decoder(noise, cond).detach().cpu()
                crps = np.absolute(pred[:100] - y_test[i][h + 1]).mean() - \
                       0.5 * np.absolute(pred[:100] - pred[100:]).mean()
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
