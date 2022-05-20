import torch.nn as nn

from models.tcn import TCN


class DeepARModel(nn.Module):
    """
    DeepAR network
    :param condition_size: size of the condition window
    :param hidden_size: hidden size of the RNN/TCN cell
    :param hidden_layer_depth: number of layers in RNN/TCN cell
    :param cell_type: GRU/LSTM/TCN to be used as a basic building block
    :param num_channel: number of channels in the TCN network
    :param num_layers: number of layers in the TCN network
    :param kernel_size: kernel size in the TCN network
    :param dropout_rate: percentage of nodes to dropout
    :param mean: mean of the data
    :param std: standard deviation of the data
    """
    def __init__(self, condition_size, hidden_size, hidden_layer_depth=2, cell_type='LSTM',
                 num_channel=25, num_layers=12, kernel_size=8, dropout_rate=0., mean=0, std=1):
        super().__init__()

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

        self.model = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.output_to_mean = nn.Linear(hidden_size, 1)
        self.output_to_logvar = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Softplus()
        )

    def forward(self, x):
        """
        :param x: condition data input, of shape (batch_size, condition_size)
        :return: output_mean: mean of the predicted distribution, of shape (batch_size, 1)
        :return: output_logvar: standard deviation of the predicted distribution, of shape (batch_size, 1)
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

        output = self.model(latent)
        output_mean = self.output_to_mean(output)
        output_logvar = self.output_to_logvar(output)
        return output_mean, output_logvar
