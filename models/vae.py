import torch
import torch.nn as nn

from models.tcn import TCN


class Encoder(nn.Module):
    """
    Encoder network
    :param condition_size: size of the condition window
    :param hidden_size: hidden size of the RNN
    :param hidden_layer_depth: number of layers in RNN
    :param dropout: percentage of nodes to dropout
    :param cell_type: GRU/LSTM to be used as a basic building block
    :param num_channel: number of channels in the TCN network
    :param num_layers: number of layers in the TCN network
    :param kernel_size: kernel size in the TCN network
    :param mean: mean of the data
    :param std: standard deviation of the data
    """
    def __init__(self, condition_size, hidden_size, hidden_layer_depth, dropout, cell_type='LSTM',
                 num_channel=25, num_layers=12, kernel_size=8, mean=0, std=1):
        super().__init__()

        self.condition_size = condition_size
        self.mean = mean
        self.std = std

        if cell_type == 'LSTM':
            self.model = nn.LSTM(input_size=1,
                                 hidden_size=hidden_size,
                                 num_layers=hidden_layer_depth,
                                 dropout=dropout)
        elif cell_type == 'GRU':
            self.model = nn.GRU(input_size=1,
                                hidden_size=hidden_size,
                                num_layers=hidden_layer_depth,
                                dropout=dropout)
        elif cell_type == 'TCN':
            channel_size = [num_channel] * num_layers
            self.model = TCN(input_size=1,
                             output_size=hidden_size,
                             num_channels=channel_size,
                             kernel_size=kernel_size,
                             dropout=dropout)
        else:
            raise NotImplementedError

    def forward(self, y, x):
        """
        :param y: future data input, of shape (batch_size, 1)
        :param x: condition data input, of shape (batch_size, condition_size)
        :return: last hidden state of encoder, of shape (batch_size, hidden_size)
        """
        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std
        data = torch.cat((y, x), dim=1)

        if isinstance(self.model, nn.LSTM):
            data = data.view(-1, self.condition_size + 1, 1).transpose(0, 1)
            _, (h_end, c_end) = self.model(data)
            encoder_output = h_end[-1, :, :]
        elif isinstance(self.model, nn.GRU):
            data = data.view(-1, self.condition_size + 1, 1).transpose(0, 1)
            _, h_end = self.model(data)
            encoder_output = h_end[-1, :, :]
        elif isinstance(self.model, TCN):
            data = data.view(-1, self.condition_size + 1, 1).transpose(1, 2)
            encoder_output = self.model(data)
        else:
            raise NotImplementedError

        return encoder_output


class Reparameterize(nn.Module):
    """
    Lambda module which converts output of encoder to latent vector
    :param hidden_size: hidden size of the encoder
    :param latent_length: length of the latent vector
    """
    def __init__(self, hidden_size, latent_length):
        super().__init__()

        self.latent_mean = None
        self.latent_logvar = None

        self.hidden_to_mean = nn.Linear(hidden_size, latent_length)
        self.hidden_to_logvar = nn.Linear(hidden_size, latent_length)

        nn.init.xavier_uniform_(self.hidden_to_mean.weight)
        nn.init.xavier_uniform_(self.hidden_to_logvar.weight)

    def forward(self, latent):
        """
        :param latent: last hidden state of encoder, of shape (batch_size, hidden_size)
        :return: latent vector, of shape (batch_size, latent_length)
        """
        self.latent_mean = self.hidden_to_mean(latent)
        self.latent_logvar = self.hidden_to_logvar(latent)

        std = torch.exp(0.5 * self.latent_logvar)
        eps = torch.randn_like(std)
        latent = eps.mul(std).add_(self.latent_mean)
        return latent


class Decoder(nn.Module):
    """
    Decoder network
    :param condition_size: size of the condition window
    :param hidden_size: hidden size of the RNN
    :param hidden_layer_depth: number of layers in RNN
    :param latent_length: length of the latent vector
    :param output_size: size of the output
    :param dtype: Depending on cuda enabled/disabled, create the tensor
    :param cell_type: GRU/LSTM to be used as a basic building block
    :param num_channel: number of channels in the TCN network
    :param num_layers: number of layers in the TCN network
    :param kernel_size: kernel size in the TCN network
    :param device: cpu/cuda to be used as a device
    :param mean: mean of the data
    :param std: standard deviation of the data
    """
    def __init__(self, condition_size, hidden_size, hidden_layer_depth, latent_length, output_size, dtype,
                 cell_type='LSTM', num_channel=25, num_layers=12, kernel_size=8, dropout=0., device='cpu', mean=0, std=1):
        super(Decoder, self).__init__()

        self.hidden_size = hidden_size
        self.condition_size = condition_size
        self.hidden_layer_depth = hidden_layer_depth
        self.dtype = dtype
        self.device = device
        self.mean = mean
        self.std = std

        self.latent_to_hidden = nn.Linear(latent_length, self.hidden_size)

        if cell_type == 'LSTM':
            self.model = nn.LSTM(input_size=1,
                                 hidden_size=hidden_size,
                                 num_layers=hidden_layer_depth)
        elif cell_type == 'GRU':
            self.model = nn.GRU(input_size=1,
                                hidden_size=hidden_size,
                                num_layers=hidden_layer_depth)
        elif cell_type == 'TCN':
            channel_size = [num_channel] * num_layers
            self.model = TCN(input_size=1,
                             output_size=hidden_size,
                             num_channels=channel_size,
                             kernel_size=kernel_size,
                             dropout=dropout)
        else:
            raise NotImplementedError

        self.hidden_to_output = nn.Linear(self.hidden_size, output_size)

        nn.init.xavier_uniform_(self.latent_to_hidden.weight)
        nn.init.xavier_uniform_(self.hidden_to_output.weight)

    def forward(self, latent, x):
        """
        :param latent: latent vector, of shape (batch_size, latent_length)
        :param x: condition data input, of shape (batch_size, condition_size)
        :return: reconstructed future data, of shape (batch_size, 1)
        """
        x = (x - self.mean) / self.std
        x = x.view(-1, self.condition_size, 1)
        batch_size = x.shape[0]

        c_0 = torch.zeros(self.hidden_layer_depth,
                          batch_size,
                          self.hidden_size,
                          requires_grad=True)\
            .type(self.dtype).to(self.device)

        h = self.latent_to_hidden(latent)

        if isinstance(self.model, nn.LSTM):
            x = x.transpose(0, 1)
            h_0 = torch.stack([h for _ in range(self.hidden_layer_depth)])
            output, _ = self.model(x, (h_0, c_0))
            output = output[-1, :, :]
        elif isinstance(self.model, nn.GRU):
            x = x.transpose(0, 1)
            h_0 = torch.stack([h for _ in range(self.hidden_layer_depth)])
            output, _ = self.model(x, h_0)
            output = output[-1, :, :]
        elif isinstance(self.model, TCN):
            h = h.view(-1, self.hidden_size, 1)
            data = torch.cat((h, x), dim=1)
            data = data.transpose(1, 2)
            output = self.model(data)
        else:
            raise NotImplementedError

        decoder_output = self.hidden_to_output(output)
        decoder_output = decoder_output * self.std + self.mean
        return decoder_output


class VAEModel(nn.Module):
    """
    Variational auto-encoder network
    :param condition_size: size of the condition window
    :param enc_hidden_size: hidden size of the RNN for Encoder network
    :param dec_hidden_size: hidden size of the RNN for Decoder network
    :param enc_hidden_layer_depth: number of layers in RNN for Encoder network
    :param dec_hidden_layer_depth: number of layers in RNN for Decoder network
    :param latent_length: length of the latent vector
    :param cell_type: GRU/LSTM to be used as a basic building block
    :param enc_dropout_rate: percentage of nodes to dropout of Encoder
    :param dec_dropout_rate: percentage of nodes to dropout of Decoder
    :param enc_num_channel: number of channels in the TCN network of Encoder
    :param dec_num_channel: number of channels in the TCN network of Decoder
    :param num_layers: number of layers in the TCN network
    :param kernel_size: size of kernel in the TCN network
    :param mean: mean of the data
    :param std: standard deviation of the data
    """
    def __init__(self, condition_size, enc_hidden_size=64, dec_hidden_size=64, enc_hidden_layer_depth=2,
                 dec_hidden_layer_depth=2, latent_length=24, cell_type='LSTM', enc_dropout_rate=0., dec_dropout_rate=0.,
                 enc_num_channel=25, dec_num_channel=25, num_layers=12, kernel_size=8, device="cpu", mean=0, std=1):
        super().__init__()

        self.encoder = Encoder(condition_size=condition_size,
                               hidden_size=enc_hidden_size,
                               hidden_layer_depth=enc_hidden_layer_depth,
                               dropout=enc_dropout_rate,
                               cell_type=cell_type,
                               num_channel=enc_num_channel,
                               num_layers=num_layers,
                               kernel_size=kernel_size,
                               mean=mean,
                               std=std)

        self.reparameterize = Reparameterize(hidden_size=enc_hidden_size,
                                             latent_length=latent_length)

        self.decoder = Decoder(condition_size=condition_size,
                               hidden_size=dec_hidden_size,
                               hidden_layer_depth=dec_hidden_layer_depth,
                               latent_length=latent_length,
                               output_size=1,
                               dtype=torch.FloatTensor,
                               dropout=dec_dropout_rate,
                               cell_type=cell_type,
                               num_channel=dec_num_channel,
                               num_layers=num_layers,
                               kernel_size=kernel_size,
                               device=device,
                               mean=mean,
                               std=std)

    def forward(self, y, x):
        """
        :param y: future data input, of shape (batch_size, condition_size)
        :param x: condition data input, of shape (batch_size, 1)
        :return: reconstructed future data, of shape (batch_size, 1)
        """
        encoder_output = self.encoder(y, x)
        latent = self.reparameterize(encoder_output)
        decoder_output = self.decoder(latent, x)

        return decoder_output
