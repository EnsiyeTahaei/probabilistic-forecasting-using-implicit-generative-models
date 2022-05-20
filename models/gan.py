import torch
import torch.nn as nn

from models.tcn import TCN


class Generator(nn.Module):
    """
    Generator network
    :param noise_dim: dimensionality of the noise space
    :param condition_size: size of the condition window
    :param gen_hidden_size: size of the cells in generator
    :param hidden_layer_depth: number of layers in RNN
    :param cell_type: GRU/LSTM to be used as a basic building block
    :param num_channel: number of channels in the TCN network
    :param num_layers: number of layers in the TCN network
    :param kernel_size: kernel size in the TCN network
    :param dropout_rate: percentage of nodes to dropout
    :param mean: mean of the data
    :param std: standard deviation of the data
    """
    def __init__(self, noise_dim, condition_size, gen_hidden_size, hidden_layer_depth=2, cell_type='LSTM',
                 num_channel=25, num_layers=12, kernel_size=8, dropout_rate=0., mean=0, std=1):
        super().__init__()
        self.condition_size = condition_size
        self.mean = mean
        self.std = std

        if cell_type == "GRU":
            self.condition_to_latent = nn.GRU(input_size=1,
                                              hidden_size=gen_hidden_size,
                                              num_layers=hidden_layer_depth)
        elif cell_type == "LSTM":
            self.condition_to_latent = nn.LSTM(input_size=1,
                                               hidden_size=gen_hidden_size,
                                               num_layers=hidden_layer_depth)
        elif cell_type == 'TCN':
            channel_size = [num_channel] * num_layers
            self.condition_to_latent = TCN(input_size=1,
                                           output_size=gen_hidden_size,
                                           num_channels=channel_size,
                                           kernel_size=kernel_size,
                                           dropout=dropout_rate)
        else:
            raise NotImplementedError

        self.model = nn.Sequential(
            nn.Linear(in_features=gen_hidden_size + noise_dim, out_features=gen_hidden_size + noise_dim),
            nn.ReLU(),
            nn.Linear(in_features=gen_hidden_size + noise_dim, out_features=1)
        )

    def forward(self, noise, x):
        """
        :param noise: noise vector, of shape (batch_size, noise_dim)
        :param x: condition data input, of shape (batch_size, condition_size)
        :return: generated data, of shape (batch_size, 1)
        """
        x = (x - self.mean) / self.std
        x = x.view(-1, self.condition_size, 1)
        if isinstance(self.condition_to_latent, TCN):
            x = x.transpose(1, 2)
            latent = self.condition_to_latent(x)
        else:
            x = x.transpose(0, 1)
            latent_, _ = self.condition_to_latent(x)
            latent = latent_[-1]

        generator_input = torch.cat((latent, noise), dim=1)

        output = self.model(generator_input)
        output = output * self.std + self.mean
        return output


class Discriminator(nn.Module):
    """
    Discriminator network
    :param condition_size: size of the condition window
    :param dis_hidden_size: size of the cells in discriminator
    :param hidden_layer_depth: number of layers in RNN
    :param cell_type: GRU/LSTM to be used as a basic building block
    :param num_channel: number of channels in the TCN network
    :param num_layers: number of layers in the TCN network
    :param kernel_size: kernel size in the TCN network
    :param dropout_rate: percentage of nodes to dropout
    :param mean: mean of the data
    :param std: standard deviation of the data
    """
    def __init__(self, condition_size, dis_hidden_size, hidden_layer_depth=2, cell_type='LSTM',
                 num_channel=25, num_layers=12, kernel_size=8, dropout_rate=0., mean=0, std=1):
        super().__init__()
        self.condition_size = condition_size
        self.mean = mean
        self.std = std

        if cell_type == "GRU":
            self.input_to_latent = nn.GRU(input_size=1,
                                          hidden_size=dis_hidden_size,
                                          num_layers=hidden_layer_depth)
        elif cell_type == "LSTM":
            self.input_to_latent = nn.LSTM(input_size=1,
                                           hidden_size=dis_hidden_size,
                                           num_layers=hidden_layer_depth)
        elif cell_type == 'TCN':
            channel_size = [num_channel] * num_layers
            self.input_to_latent = TCN(input_size=1,
                                       output_size=dis_hidden_size,
                                       num_channels=channel_size,
                                       kernel_size=kernel_size,
                                       dropout=dropout_rate)
        else:
            raise NotImplementedError

        self.model = nn.Sequential(
            nn.Linear(in_features=dis_hidden_size, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, y, x):
        """
        :param y: data input, of shape (batch_size, 1)
        :param x: condition data input, of shape (batch_size, condition_size)
        :return: classification result vector, of shape (batch_size, 1)
        """
        discriminator_input = torch.cat((x, y.view(-1, 1)), dim=1)
        discriminator_input = (discriminator_input - self.mean) / self.std
        discriminator_input = discriminator_input.view(-1, self.condition_size + 1, 1)

        if isinstance(self.input_to_latent, TCN):
            discriminator_input = discriminator_input.transpose(1, 2)
            discriminator_latent = self.input_to_latent(discriminator_input)
        else:
            discriminator_input = discriminator_input.transpose(0, 1)
            discriminator_latent, _ = self.input_to_latent(discriminator_input)
            discriminator_latent = discriminator_latent[-1]

        output = self.model(discriminator_latent)
        return output
