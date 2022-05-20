import os
import pickle
import statistics
import time

import numpy as np
import torch
import torch.optim as optim
from scipy.stats import norm

import utils
from models.deepar import DeepARModel


class NormalNLLLoss:
    """
    Calculation the negative log likelihood of normal distribution
    """
    def __call__(self, mu, sigma, gt):
        eps = 1e-16
        dist = torch.distributions.normal.Normal(mu, sigma + eps)
        ll = dist.log_prob(gt)
        nll = -torch.mean(ll)
        return nll


class DeepAR(object):
    """
    DeepAR model
    :param config: config information
    :param dataset_x: train dataset
    :param dataset_y: test dataset
    """
    def __init__(self, config, dataset_x, dataset_y):
        # Optimization parameters
        self.max_steps = config['max_steps']
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
        self.dAR_hidden_size = config['dAR_hidden_size']
        self.dAR_hidden_depth = config['dAR_hidden_depth']
        self.dAR_num_channel = config['dAR_num_channel']
        self.dAR_dropout_rate = config['dAR_dropout_rate']
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
        self.deepAR = DeepARModel(condition_size=self.condition_size,
                                  hidden_size=self.dAR_hidden_size,
                                  hidden_layer_depth=self.dAR_hidden_depth,
                                  cell_type=self.cell_type,
                                  num_channel=self.dAR_num_channel,
                                  num_layers=self.num_layers,
                                  kernel_size=self.kernel_size,
                                  dropout_rate=self.dAR_dropout_rate,
                                  mean=self.data_mean,
                                  std=self.data_std).to(self.device)
        self.optimizer = getattr(optim, self.optimizer_name)(self.deepAR.parameters(), lr=self.lr)
        self.NLLLoss = NormalNLLLoss()
        # self.scheduler = ReduceLROnPlateau(optimizer=self.optimizer,
        #                                    mode='min',
        #                                    patience=20,
        #                                    verbose=True)

        # Print networks architecture
        print("***********************************")
        print("*****  Networks architecture  *****")
        print(self.deepAR)
        print("***********************************")

    def train(self):
        print("*********  Training model  ********")

        self.training_info['loss'] = []
        self.training_info['loss_val'] = []
        self.training_info['per_step_time'] = []
        self.training_info['total_time'] = []

        x_train = torch.tensor(self.dataset_x["train"], device=self.device, dtype=torch.float32)
        y_train = torch.tensor(self.dataset_y["train"], device=self.device, dtype=torch.float32)
        x_val = torch.tensor(self.dataset_x["val"], device=self.device, dtype=torch.float32)
        y_val = torch.tensor(self.dataset_y["val"], device=self.device, dtype=torch.float32)
        val_mean = self.dataset_x["val"].mean()
        val_std = self.dataset_x["val"].std()
        norm_y_val = (y_val - val_mean) / val_std
        best_nll = np.inf
        no_improve, stop = 0, False

        start_time = time.time()

        for step in range(1, self.max_steps + 1):
            step_start_time = time.time()

            idx = self.rs.choice(x_train.shape[0], self.batch_size)
            condition = x_train[idx]
            real_data = y_train[idx]
            norm_real_data = (real_data - self.data_mean) / self.data_std

            self.deepAR.zero_grad()
            p_mean, p_sigma = self.deepAR(condition)
            loss = self.NLLLoss(p_mean, p_sigma, norm_real_data)
            loss.backward()
            self.optimizer.step()

            # Validation of the model

            with torch.no_grad():
                p_mean, p_sigma = self.deepAR(x_val)
            loss_val = self.NLLLoss(p_mean, p_sigma, norm_y_val)

            if 200 < step and loss_val <= best_nll and loss_val != np.inf:
                best_nll = loss_val
                print("-- Step {}: !!! New best model !!! CRPS {}".format(step, loss_val))
                torch.save({'deepAR_state_dict': self.deepAR.state_dict()}, self.result_dir + "/best_model")
                no_improve = 0
            else:
                no_improve += 1
                if no_improve == self.patience:
                    stop = True

            # Logging progress
            if step % self.log_interval == 0:
                print("-- [Step {}/{}]\t[deepAR loss: {}]\t[val loss: {}]".format(step, self.max_steps, loss, loss_val))

            self.save_model(step=step)
            # self.save_info(info_label="training")
            step_end_time = time.time()
            step_time = step_end_time - step_start_time
            self.training_info['per_step_time'].append(step_time)
            self.training_info['loss'].append(loss)
            self.training_info['loss_val'].append(loss_val)

            # self.scheduler.step(crps)

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

        eps = 1e-16
        for _ in range(200):
            with torch.no_grad():
                p_mean, p_sigma = self.deepAR(x_test)
                pred = []
                for idx in range(x_test.size(0)):
                    sample = torch.normal(mean=p_mean[idx][0],
                                          std=p_sigma[idx][0] + eps,
                                          size=(1,),
                                          device=self.device,
                                          dtype=torch.float32)
                    sample = sample * self.data_std + self.data_mean
                    pred.append(sample.detach().cpu().numpy().flatten())
                pred = np.vstack(pred).flatten()
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
        self.load()

        preds = []
        eps = 1e-16
        for _ in range(200):
            with torch.no_grad():
                p_mean, p_sigma = self.deepAR(x_test)
                pred = []
                for idx in range(x_test.size(0)):
                    sample = torch.normal(mean=p_mean[idx][0],
                                          std=p_sigma[idx][0] + eps,
                                          size=(1,),
                                          device=self.device,
                                          dtype=torch.float32)
                    sample = sample * self.data_std + self.data_mean
                    pred.append(sample.detach().cpu().numpy().flatten())
                pred = np.vstack(pred).flatten()
                preds.append(pred)
        preds = np.vstack(preds)
        preds = preds.flatten()
        return preds

    def save_model(self, step):
        torch.save(self.deepAR.state_dict(),
                   self.models_dir + "/deepAR_parameters_" + str(step) + ".pth")

    def save_info(self, info_label):
        if info_label == "training":
            with open(self.result_dir + '/training_info.pkl', 'wb') as f:
                pickle.dump(self.training_info, f)
        elif info_label == "testing":
            with open(self.result_dir + '/testing_info.pkl', 'wb') as f:
                pickle.dump(self.testing_info, f)

    def load(self):
        checkpoint = torch.load(self.result_dir + "/best_model", map_location=torch.device('cpu'))
        self.deepAR.load_state_dict(checkpoint['deepAR_state_dict'])

    def forecast(self):
        forecast_dir = os.path.join(self.result_dir, "forecasting")
        forecast_dir = os.path.join(forecast_dir, str(self.quantile))
        utils.check_dir(forecast_dir)

        x_test = torch.tensor(self.dataset_x["test"], device=self.device, dtype=torch.float32)
        y_test = torch.tensor(self.dataset_y["test"], dtype=torch.float32)
        self.load()
        total_preds = []
        crps_scores = []

        eps = 1e-16
        for i, cond in enumerate(x_test):
            cond = cond.view(1, -1).repeat(self.sample_size, 1)
            with torch.no_grad():
                p_mean, p_sigma = self.deepAR(cond)
                pred = []
                for idx in range(self.sample_size):
                    sample = torch.normal(mean=p_mean[idx][0],
                                          std=p_sigma[idx][0] + eps,
                                          size=(1,),
                                          device=self.device,
                                          dtype=torch.float32)
                    sample = sample * self.data_std + self.data_mean
                    pred.append(sample.detach().cpu())
            pred = torch.tensor(pred, dtype=torch.float32).view(-1, 1)

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
                with torch.no_grad():
                    p_mean, p_sigma = self.deepAR(cond)
                    pred = []
                    for idx in range(self.sample_size):
                        sample = torch.normal(mean=p_mean[idx][0],
                                              std=p_sigma[idx][0] + eps,
                                              size=(1,),
                                              device=self.device,
                                              dtype=torch.float32)
                        sample = sample * self.data_std + self.data_mean
                        pred.append(sample.detach().cpu())
                pred = torch.tensor(pred, dtype=torch.float32)

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
