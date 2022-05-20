import torch
import torch.nn as nn

from models.tcn import TCN


class Encoder(nn.Module):
    """
    Encoder network
    :param condition_size: size of the condition window
    :param hidden_size: hidden size of the RNN
    :param hidden_layer_depth: number of layers in RNN
    :param latent_length: length of the latent vector
    :param cell_type: GRU/LSTM to be used as a basic building block
    :param num_channel: number of channels in the TCN network
    :param num_layers: number of layers in the TCN network
    :param kernel_size: kernel size in the TCN network
    :param dropout_rate: percentage of nodes to dropout
    :param mean: mean of the data
    :param std: standard deviation of the data
    """
    def __init__(self, condition_size, hidden_size, latent_length, hidden_layer_depth=2, cell_type='LSTM',
                 num_channel=25, num_layers=12, kernel_size=8, dropout_rate=0., mean=0, std=1):
        super().__init__()

        self.condition_size = condition_size
        self.mean = mean
        self.std = std

        if cell_type == 'LSTM':
            self.condition_to_latent = nn.LSTM(input_size=1,
                                               hidden_size=hidden_size,
                                               num_layers=hidden_layer_depth,
                                               dropout=dropout_rate)
        elif cell_type == 'GRU':
            self.condition_to_latent = nn.GRU(input_size=1,
                                              hidden_size=hidden_size,
                                              num_layers=hidden_layer_depth,
                                              dropout=dropout_rate)
        elif cell_type == 'TCN':
            channel_size = [num_channel] * num_layers
            self.condition_to_latent = TCN(input_size=1,
                                           output_size=hidden_size,
                                           num_channels=channel_size,
                                           kernel_size=kernel_size,
                                           dropout=dropout_rate)
        else:
            raise NotImplementedError

        self.prediction_to_latent = nn.Linear(1, hidden_size)
        self.model = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size * 4),
            nn.ReLU(),
            nn.Linear(hidden_size * 4, latent_length),
        )

    def forward(self, y, x):
        """
        :param y: future data input, of shape (batch_size, 1)
        :param x: condition data input, of shape (batch_size, condition_size)
        :return: latent vector, of shape (batch_size, latent_length)
        """
        x = (x - self.mean) / self.std
        x = x.view(-1, self.condition_size, 1)
        if isinstance(self.condition_to_latent, TCN):
            x = x.transpose(1, 2)
            latent1 = self.condition_to_latent(x)
        else:
            x = x.transpose(0, 1)
            latent_, _ = self.condition_to_latent(x)
            latent1 = latent_[-1]

        y = (y - self.mean) / self.std
        y = y.view(-1, 1)
        latent2 = self.prediction_to_latent(y)

        latent_encoder = torch.cat((latent1, latent2), dim=1)
        output_encoder = self.model(latent_encoder)
        return output_encoder


class Decoder(nn.Module):
    """
    Decoder network
    :param condition_size: size of the condition window
    :param hidden_size: hidden size of the RNN
    :param hidden_layer_depth: number of layers in RNN
    :param latent_length: length of the latent vector
    :param cell_type: GRU/LSTM to be used as a basic building block
    :param num_channel: number of channels in the TCN network
    :param num_layers: number of layers in the TCN network
    :param kernel_size: kernel size in the TCN network
    :param dropout_rate: percentage of nodes to dropout
    :param mean: mean of the data
    :param std: standard deviation of the data
    """
    def __init__(self, condition_size, hidden_size, latent_length, hidden_layer_depth=2, cell_type='LSTM',
                 num_channel=25, num_layers=12, kernel_size=8, dropout_rate=0., mean=0, std=1):
        super(Decoder, self).__init__()

        self.condition_size = condition_size
        self.mean = mean
        self.std = std

        if cell_type == 'LSTM':
            self.condition_to_latent = nn.LSTM(input_size=1,
                                               hidden_size=hidden_size,
                                               num_layers=hidden_layer_depth)
        elif cell_type == 'GRU':
            self.condition_to_latent = nn.GRU(input_size=1,
                                              hidden_size=hidden_size,
                                              num_layers=hidden_layer_depth)
        elif cell_type == 'TCN':
            channel_size = [num_channel] * num_layers
            self.condition_to_latent = TCN(input_size=1,
                                           output_size=hidden_size,
                                           num_channels=channel_size,
                                           kernel_size=kernel_size,
                                           dropout=dropout_rate)
        else:
            raise NotImplementedError

        self.latent_to_latent = nn.Linear(latent_length, hidden_size)
        self.model = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, latent, x):
        """
        :param latent: latent vector, of shape (batch_size, latent_length)
        :param x: condition data input, of shape (batch_size, condition_size)
        :return: reconstructed future data, of shape (batch_size, 1)
        """
        x = (x - self.mean) / self.std
        x = x.view(-1, self.condition_size, 1)
        if isinstance(self.condition_to_latent, TCN):
            x = x.transpose(1, 2)
            latent1 = self.condition_to_latent(x)
        else:
            x = x.transpose(0, 1)
            latent_, _ = self.condition_to_latent(x)
            latent1 = latent_[-1]

        latent2 = self.latent_to_latent(latent)

        latent_decoder = torch.cat((latent1, latent2), dim=1)

        output_decoder = self.model(latent_decoder)
        output_decoder = output_decoder * self.std + self.mean
        return output_decoder


class Discriminator(nn.Module):
    """
    Discriminator network
    :param input_size: input size of the discriminator network
    """
    def __init__(self, input_size):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, z):
        """
        :param z: data input, of shape (batch_size, latent_length)
        :return: classification result vector, of shape (batch_size, 1)
        """
        output_discriminator = self.model(z)
        return output_discriminator
