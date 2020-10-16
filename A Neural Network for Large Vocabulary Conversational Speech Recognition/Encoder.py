import torch
from torch import nn
from torch.nn.utils.rnn import *
from dropout import *


class Encoder(nn.Module):
    def __init__(self, opt):
        super(Encoder, self).__init__()

        self.num_layers = opt.encoder_num_layers
        self.blstm = []
        hidden_dim = opt.encoder_hidden_dim
        self.blstm.append(nn.LSTM(input_size = opt.input_dim, hidden_size = hidden_dim, num_layers = 1, bias = False, bidirectional = opt.is_bidirectional))
        for layer in range(1, self.num_layers):
            self.blstm.append(nn.LSTM(input_size = hidden_dim * 4, hidden_size = hidden_dim, num_layers = 1, bias = False, bidirectional = opt.is_bidirectional))

        self.blstm = torch.nn.ModuleList(self.blstm)

        self.dropout =  LockedDropout(dropout = opt.dropout)

        

    def forward(self, inputs, lens):

        rnn_inp = pack_padded_sequence(inputs, lengths = lens, enforce_sorted = False)

        # the shape of output (T, N, hidden * 2)
        outputs, _ = self.blstm[0](rnn_inp)

        for layer in range(1, self.num_layers):

            outputs, _ = pad_packed_sequence(outputs)

            # permute the output to the shape of (N, T, hiddens)
            outputs = outputs.permute(1, 0, 2)
            if outputs.size(1) % 2 == 1:
                outputs = outputs[:, :-1, :].reshape(outputs.size(0), outputs.size(1) // 2, outputs.size(2) * 2)
            else:
                outputs = outputs.reshape(outputs.size(0), outputs.size(1) // 2, outputs.size(2) * 2)

            # permute the output back to the shape of (T, N, hiddens)
            outputs = outputs.permute(1, 0, 2)
            outputs = self.dropout(outputs)
            # pack the output
            lens = lens // 2
            outputs = pack_padded_sequence(outputs, lengths = lens, enforce_sorted = False)

            outputs, hidden = self.blstm[layer](outputs)
            
        outs, lens = pad_packed_sequence(outputs)

        return outs, lens, hidden

class SimpleEncoder(nn.Module):
    def __init__(self, opt):
        super(SimpleEncoder, self).__init__()
        # super(SimpleEncoder, self).__int__()
        self.lstm = nn.LSTM(input_size = opt.input_dim, hidden_size = opt.encoder_hidden_dim, num_layers = 1, bias = False, bidirectional = opt.is_bidirectional)

    def forward(self, x, lens):

        rnn_inp = pack_padded_sequence(x, lengths = lens, batch_first = False, enforce_sorted = False)
        outputs, hidden = self.lstm(rnn_inp)
        outputs, _ = pad_packed_sequence(outputs)

        return outputs, hidden


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups = 1):
        super(BasicBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size = kernel_size, stride = stride, padding = padding, bias = False, groups = groups)
        self.bn = nn.BatchNorm1d(out_channels)
#         self.drop = nn.Dropout(p = 0.2)
        self.tanh = nn.Hardtanh(inplace = True)

        
    def forward(self, inputs):
        outputs = self.conv(inputs)
        outputs = self.bn(outputs)
#         outputs = self.drop(outputs)
        outputs = self.tanh(outputs)
        return outputs

class ConvEncoder(nn.Module):
    def __init__(self, opt):
        super(ConvEncoder, self).__init__()

        self.num_layers = opt.encoder_num_layers
        self.cnn1 = BasicBlock(opt.input_dim, opt.encoder_hidden_dim, kernel_size = 3, stride = 2, padding = 0)
        self.cnn2 = BasicBlock(opt.encoder_hidden_dim // 2, opt.encoder_hidden_dim, kernel_size = 3, stride = 2, padding = 0)

        self.lstm = nn.LSTM(input_size = opt.encoder_hidden_dim, hidden_size = opt.encoder_hidden_dim, num_layers = 3, bias = False, bidirectional = opt.is_bidirectional)

        self.dropout =  LockedDropout(dropout = opt.dropout)

    def forward(self, inputs, lens):

        inputs = self.cnn1(inputs) 
        inputs = self.cnn2(inputs) 
        inputs = inputs.permute(2, 0, 1)
        lens = ((lens - 1) // 2 - 1) // 2
        rnn_inp = pack_padded_sequence(inputs, lengths = lens, enforce_sorted = False)
        
        outputs, _ = self.lstm(rnn_inp)

        # outputs = self.dropout(outputs)
            
        outs, lens = pad_packed_sequence(outputs)

        return outs, lens

