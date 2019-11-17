import torch
from torch import nn
from torch.nn.utils.rnn import *

class Encoder(nn.Module):
  def __init__(self, inp_dim, hidden_dim, embed_dim, num_layers = 3, value_size=128,key_size=128):
	super(Encoder, self).__init__()
	# self.lstm = nn.LSTM(input_size=input_dim,hidden_size=hidden_dim,num_layers=1,bidirectional=True)
	#Here you need to define the blocks of pBLSTMs


	self.num_layers = num_layers
	self.blstm = []
	self.blstm.append(nn.LSTM(input_size = inp_dim, hidden_size = hidden_dim, num_layers = 1, bias = False, bidirectional = True))
	for layer in range(1, num_layers):
		hidden_dim = hidden_dim * 2
		self.blstm.append(nn.LSTM(input_size = hidden_dim, hidden_size = hidden_dim, num_layers = 1, bias = False, bidirectional = True))

	self.pooling = []
	for layer in range(1, num_layers):
		self.pooling.append(nn.AvgPool1d(kernel_size = 2, stride = 2))

	self.key_network = nn.Linear(hidden_dim * 2**(num_layers - 1), value_size)
	self.value_network = nn.Linear(hidden_dim * 2**(num_layers - 1), key_size)
  
  def forward(self,x, lens):
	rnn_inp = pack_padded_sequence(x, lengths = lens, batch_first = False, enforce_sorted = False)

	# the shape of output (T, N, hidden * 2)
	outputs, _ = self.blstm[0](rnn_inp)

	for layer in range(1, self.num_layers):

		outputs, _ = pad_packed_sequence(outputs)
		# permute the output to the shape of (N, hidden, T)
		outputs = outputs.permute(1, 2, 0)

		# pooling to make the pyramidal structure
		outputs = self.pooling[layer](outputs)

		# permute the output back to the shape of (T, N, hidden)
		outputs = outputs.permute(2, 0, 1)
		
		# pack the output
		outputs = pack_padded_sequence(outputs, lengths = lens // 2, batch_first = False, enforce_sorted = False)

		outputs, _ = self.blstm[layer](outputs)

	linear_inputs, lens = pad_packed_sequence(outputs)

	keys = self.key_network(linear_inputs)
	value = self.value_network(linear_inputs)

	return keys, value, lens