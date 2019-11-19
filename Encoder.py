import torch
from torch import nn
from torch.nn.utils.rnn import *

class Encoder(nn.Module):
    def __init__(self, opt):
        super(Encoder, self).__init__()

        self.num_layers = opt.encoder_num_layers
        self.blstm = []
        hidden_dim = opt.encoder_hidden_dim
        self.blstm.append(nn.LSTM(input_size = opt.input_dim, hidden_size = hidden_dim, num_layers = 1, bias = False, bidirectional = opt.is_bidirectional))
        for layer in range(1, self.num_layers):
            self.blstm.append(nn.LSTM(input_size = hidden_dim * 2, hidden_size = hidden_dim, num_layers = 1, bias = False, bidirectional = opt.is_bidirectional))

        self.blstm = torch.nn.ModuleList(self.blstm)

        self.pooling = []
        for layer in range(1, self.num_layers):
            self.pooling.append(nn.AvgPool1d(kernel_size = 2, stride = 2))

        self.pooling = nn.ModuleList(self.pooling)

        self.key_network = nn.Linear(hidden_dim * 2, opt.value_size)
        self.value_network = nn.Linear(hidden_dim * 2, opt.key_size)


    def forward(self, x, lens):
        rnn_inp = pack_padded_sequence(x, lengths = lens, batch_first = False, enforce_sorted = False)

        # the shape of output (T, N, hidden * 2)
        outputs, _ = self.blstm[0](rnn_inp)

        for layer in range(1, self.num_layers):

            outputs, _ = pad_packed_sequence(outputs)
            # permute the output to the shape of (N, hidden, T)
            outputs = outputs.permute(1, 2, 0)

            # pooling to make the pyramidal structure
            outputs = self.pooling[layer - 1](outputs)

            # permute the output back to the shape of (T, N, hidden)
            outputs = outputs.permute(2, 0, 1)
            # pack the output
            lens = lens // 2
            outputs = pack_padded_sequence(outputs, lengths = lens, batch_first = False, enforce_sorted = False)

            outputs, _ = self.blstm[layer](outputs)

        linear_inputs, lens = pad_packed_sequence(outputs)

        keys = self.key_network(linear_inputs)
        value = self.value_network(linear_inputs)

        return keys, value, lens