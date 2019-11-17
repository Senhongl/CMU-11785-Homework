from Atten import *, Attention
from torch import nn
import torch

class Decoder(nn.Module):
	def __init__(self, vocab_size, hidden_dim, value_size=128, key_size=128, isAttended=False):
		super(Decoder, self).__init__()
		self.embedding = nn.Embedding(vocab_size, hidden_dim)

		self.lstm1 = nn.LSTMCell(input_size = hidden_dim + value_size, hidden_size = hidden_dim)
		self.lstm2 = nn.LSTMCell(input_size = hidden_dim, hidden_size = key_size)
		self.isAttended = isAttended
		if(isAttended):
			self.attention = Attention()
		self.character_prob = nn.Linear(key_size+value_size,vocab_size)

	def forward(self, key, values, text=None, train=True):
		'''
		:param key :(T,N,key_size) Output of the Encoder Key projection layer
		:param values: (T,N,value_size) Output of the Encoder Value projection layer
		:param text: (N,text_len) Batch input of text with text_length
		:param train: Train or eval mode
		:return predictions: Returns the character perdiction probability 
		'''
		batch_size = key.shape[1]
		if(train):
			max_len =  text.shape[1]
			embeddings = self.embedding(text)
		else:
			max_len = 250

		predictions = []
		hidden_states = [None, None]
		prediction = torch.zeros(batch_size,1).to(device)
		for i in range(max_len):
			'''
			Here you should implement Gumble noise and teacher forcing techniques
			'''
			if(train):
			char_embed = embeddings[:,i,:]
			else:
			char_embed = self.embedding(prediction.argmax(dim=-1))
			
			#When attention is True you should replace the values[i,:,:] with the context you get from attention
			
			inp = torch.cat([char_embed,values[i,:,:]], dim=1)
			hidden_states[0] = self.lstm1(inp,hidden_states[0])
			
			inp_2 = hidden_states[0][0]
			hidden_states[1] = self.lstm2(inp_2,hidden_states[1])

			output = hidden_states[1][0]
			prediction = self.character_prob(torch.cat([output, values[i,:,:]], dim=1))
			predictions.append(prediction.unsqueeze(1))

		return torch.cat(predictions, dim=1)