import torch
from torch import nn
from torch.nn.utils.rnn import *
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hidden_dim, key_size):
		super(Attention, self).__init__()
		self.linear = nn.Linear(hidden_dim, key_size)

    def forward(self, key, query, value, lengths):
		"""
		:param query: (N, H), decoder state of a single timestep
		:param key: (N, T, key_size), Key Projection from Encoder per time step
		:param value: (N, T, value_size), Value Projection from Encoder per time step
		:param lengths: (N,), lengths of source sequences
		:returns: (N, H) attended source context, and (N, T) attention vectors
		"""
		# convert query from (N, H) -> (N, key_size)
		query = self.linear(query)
		# Input/output shape of bmm: (N, T, key_size), (N, key_size, 1) -> (N, key_size, 1)
		attention = torch.bmm(key, query.unsqueeze(2)).squeeze(2)

		# Create an (N, T) boolean mask for all padding positions
		# Make use of broadcasting: (1, T), (N, 1) -> (N, T)
		mask = torch.arange(query.size(1)).unsqueeze(0) >= lengths.unsqueeze(1)
		# Set attention logits at padding positions to negative infinity.
		attention.masked_fill_(mask, -1e9)
		# Take softmax over the "source length" dimension.
		attention = nn.functional.softmax(attention, dim=1)

		# Compute attention-weighted sum of context vectors
		# Input/output shape of bmm: (N, 1, T), (N, T, value_size) -> (N, 1, value_size)
		out = torch.bmm(attention.unsqueeze(1), value).squeeze(1)
		return out, attention
