import os
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import torch


# Configuring arguments
def prepare_parser():
    parser = ArgumentParser(description="Pytorch implementation of auto-encoders for probabilistic forecasting")

    #  Optimization  #
    parser.add_argument("--max_steps", type=int, default=50000, help="Max number of steps to run")
    parser.add_argument("--dis_max_iter", type=int, default=2, help="Max number of training iteration of GAN")
    parser.add_argument("--ae_max_iter", type=int, default=4, help="Max number of training iteration of AE")
    parser.add_argument("--patience", type=int, default=5000, help="Max number of steps for early stopping")
    parser.add_argument("--batch_size", type=int, default=1000, help="Size of batch")
    parser.add_argument("--condition_size", type=int, default=24, help="Size of the condition window")
    parser.add_argument("--horizon", type=int, default=1, help="Number of ahead steps for predicting")
    parser.add_argument("--optimizer_name", type=str, default="Adam", choices=["Adam", "RMSprop", "SGD"],
                        help="Name of the optimizer")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for the optimizers")
    parser.add_argument("--b1", type=float, default=0.5, help="Hyper-parameter b1 for the optimizers")
    parser.add_argument("--b2", type=float, default=0.999, help="Hyper-parameter b2 for the optimizers")
    parser.add_argument("--quantile", type=int, default=1, help="Number of quantiles for forecasting")
    parser.add_argument("--sample_size", type=int, default=200, help="Number of samples")

    #  Networks: AAE, DeepAR, GAN, VAE #
    parser.add_argument("--model_name", type=str, default="AAE", choices=["GAN", "AAE", "VAE", "DeepAR"],
                        help="Name of the generative models")
    parser.add_argument("--cell_type", type=str, default="LSTM", choices=["GRU", "LSTM", "TCN"], help="Type of cells")
    parser.add_argument("--noise_dim", type=int, default=32, help="Dimensionality of the noise space")
    parser.add_argument("--latent_length", type=int, default=24, help="Dimensionality of the code space")
    parser.add_argument("--gen_hidden_size", type=int, default=8, help="Size of the latent of generator")
    parser.add_argument("--gen_hidden_depth", type=int, default=2, help="number of layers in RNN of generator")
    parser.add_argument("--gen_dropout_rate", type=float, default=0., help="Dropout rate of generator")
    parser.add_argument("--gen_num_channel", type=int, default=25, help="Number of channels in TCN of generator")
    parser.add_argument("--dis_hidden_size", type=int, default=64, help="Size of the latent of discriminator")
    parser.add_argument("--dis_hidden_depth", type=int, default=2, help="number of layers in RNN of discriminator")
    parser.add_argument("--dis_dropout_rate", type=float, default=0., help="Dropout rate of discriminator")
    parser.add_argument("--dis_num_channel", type=int, default=25, help="Number of channels in TCN of discriminator")
    parser.add_argument("--enc_hidden_size", type=int, default=90, help="hidden size of the RNN of encoder")
    parser.add_argument("--enc_hidden_depth", type=int, default=2, help="number of layers in RNN of encoder")
    parser.add_argument("--enc_dropout_rate", type=float, default=0., help="Dropout rate of encoder")
    parser.add_argument("--enc_num_channel", type=int, default=25, help="Number of channels in TCN network of encoder")
    parser.add_argument("--dec_hidden_size", type=int, default=90, help="hidden size of the RNN of decoder")
    parser.add_argument("--dec_hidden_depth", type=int, default=2, help="number of layers in RNN of decoder")
    parser.add_argument("--dec_dropout_rate", type=float, default=0., help="Dropout rate of decoder")
    parser.add_argument("--dec_num_channel", type=int, default=25, help="Number of channels in TCN network of decoder")
    parser.add_argument("--dAR_hidden_size", type=int, default=8, help="Size of the latent of DeepAR")
    parser.add_argument("--dAR_hidden_depth", type=int, default=2, help="number of layers in RNN of DeepAR")
    parser.add_argument("--dAR_dropout_rate", type=float, default=0., help="Dropout rate of DeepAR")
    parser.add_argument("--dAR_num_channel", type=int, default=25, help="Number of channels in TCN of DeepAR")
    parser.add_argument("--num_layers", type=int, default=12, help="Number of layers in the TCN network")
    parser.add_argument("--kernel_size", type=int, default=8, help="Size of kernel in the TCN network")

    #  Dataset  #
    parser.add_argument("--dataset_dir", type=str, default="./datasets", help="Directory for loading dataset")
    parser.add_argument("--dataset_name", type=str, default="lorenz", choices=["lorenz", "itd", "gp", "hepc"],
                        help="Name of the datasets")
    parser.add_argument("--val_rate", type=float, default=0.1, help="Split rate for validation data")
    parser.add_argument("--test_rate", type=float, default=0.2, help="Split rate for test data")

    #  Evaluation  #
    parser.add_argument("--hist_bins", type=int, default=80, help="Number of histogram bins for calculating KLD")
    parser.add_argument("--hist_min", type=float, default=-11, help="Min range of histogram for calculating KLD")
    parser.add_argument("--hist_max", type=float, default=11, help="Max range of histogram for calculating KLD")

    #  Visualization  #
    parser.add_argument("--result_dir", type=str, default="/netscratch/tahaei/thesis",
                        help="Directory for saving results")
    parser.add_argument("--log_interval", type=int, default=1, help="Number of interval steps to log")
    parser.add_argument("--save_interval", type=int, default=1, help="Number of interval steps to save information")

    #  Setup  #
    parser.add_argument("--mode", type=int, default=2, help="Tuning: 0, Training: 1, Forecasting: 2")
    parser.add_argument("--seed", type=int, default=200, help="Random Seed")
    parser.add_argument("--device", type=str, default="cpu", help="Existence of cuda device")
    parser.add_argument("--max_device", type=int, default=1, help="Max number of available cuda device")
    parser.add_argument("--process_per_device", type=int, default=2, help="Max number of processes on a cuda device")
    parser.add_argument("--tune_cell", type=str, default="rnn", choices=["RNN", "TCN"], help="Type of cells for tuning")

    return check_parser(parser.parse_args())


# Checking arguments
def check_parser(parser):
    parser.result_dir = os.path.join(parser.result_dir, parser.dataset_name)
    parser.result_dir = os.path.join(parser.result_dir, parser.model_name)
    check_dir(parser.result_dir)

    assert parser.max_steps >= 1
    assert parser.dis_max_iter >= 1
    assert parser.ae_max_iter >= 1
    assert parser.batch_size >= 1
    assert parser.condition_size >= 1
    assert parser.horizon >= 1
    assert parser.noise_dim >= 1
    assert parser.latent_length >= 1

    parser.seed = check_seed(parser.seed)
    parser.device = check_device(parser.mode)

    print("*******  Hyper-parameters  ********")
    for k, v in vars(parser).items():
        print("{}:\t{}".format(k, v))
    print("***********************************")

    return parser


# Checking directory
def check_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    return dir_name


# Checking random seed
def check_seed(seed):
    torch.manual_seed(seed)
    return seed


# Checking device
def check_device(mode):
    device = "cpu" if not torch.cuda.is_available() else "cuda"
    return device


def load_quantiles(quantile, sample_size):
    q_factors = []
    q = 1 / (quantile + 1)
    for counter in range(quantile):
        q_factors.append(q * (counter + 1))
    q_sizes = [int(sample_size / quantile)] * (quantile - 1)
    q_sizes.append(sample_size - sum(q_sizes))
    return q_factors, q_sizes


def visualize_predictions(predictions, ground_truth, title, bins, range_min, range_max, destination, condition=None):
    plt.figure(figsize=(12, 8))
    plt.hist(ground_truth, bins=bins, density=True, range=(range_min, range_max), color="#80c680", edgecolor="r",
             label="ground truth", alpha=0.6)
    plt.hist(predictions, bins=bins, density=True, range=(range_min, range_max), color="#eeadda", edgecolor="r",
             label="predictions", alpha=0.4)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("p(x)")
    plt.legend()
    if condition is not None:
        if "randomly" not in title:
            plt_name = "Condition" + str(condition)
        else:
            plt_name = "Condition" + str(condition) + "_RandomTimeWindow"
    else:
        plt_name = "EntireTestSet"
    plt.savefig(destination + "/" + plt_name + ".png")
    plt.close()


def visualize_crps(predictions, ground_truth, cond_idx, horizon, title, destination):
    means = predictions.mean(axis=0)
    stds = predictions.std(axis=0)
    plt.figure(figsize=(12, 8))
    plt.plot(ground_truth, label="ground truth", color='royalblue')
    plt.plot(means, label="median predictions", color='salmon')
    positions = tuple(i for i in range(0, horizon))
    plt.fill_between(positions, means - stds, means + stds, label="50% confidence interval",
                     alpha=0.4, color='lightsalmon')
    plt.fill_between(positions, means - (2 * stds), means + (2 * stds), label="90% confidence interval",
                     alpha=0.2, color='lightsalmon')
    labels = tuple("t+" + str(i) for i in range(1, horizon + 1))
    plt.xticks(positions, labels)
    plt.xlabel("time steps")
    plt.ylabel("Predictions")
    plt.title(title)
    plt.legend()
    plt_name = "Predictions_" + str(cond_idx)
    plt.savefig(destination + "/" + plt_name + ".png")
    plt.close()
