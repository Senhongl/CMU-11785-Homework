from Atten import *
from torch import nn
import torch
import random
from torch.distributions.gumbel import Gumbel

class Decoder(nn.Module):
    def __init__(self, opt):
        super(Decoder, self).__init__()
        self.opt = opt
        self.embedding = nn.Embedding(opt.vocab_size, opt.embedding_size)
        self.attention = Attention(opt)
        self.gumbel = Gumbel(torch.tensor([0.0]), torch.tensor([1.0]))
        self.lstm1 = nn.LSTMCell(input_size = opt.embedding_size + opt.value_size, hidden_size = opt.decoder_hidden_dim)
        self.lstm2 = nn.LSTMCell(input_size = opt.decoder_hidden_dim, hidden_size = opt.key_size)

        self.character_prob = nn.Linear(opt.key_size + opt.value_size, opt.vocab_size)

    def forward(self, key, values, text = None, lens = None, mode = 'train'):
        '''
        :param key :(N, T, key_size) Output of the Encoder Key projection layer
        :param values: (N, T, value_size) Output of the Encoder Value projection layer
        :param text: (N, text_len) Batch input of text with text_length
        :param lens: (N, ) Batch input of sequence length
        :return predictions: Returns the character perdiction probability 
        '''

        if mode == 'train':
            max_len =  text.shape[1]
            # the shape of embeddings would be (N, text_len, embed_size)
            embeddings = self.embedding(text)
            prediction = torch.zeros(self.opt.train_batch_size, 1).to(self.opt.device)
        elif mode == 'val':
            max_len =  text.shape[1]
            # the shape of embeddings would be (N, text_len, embed_size)
            embeddings = self.embedding(text)
            prediction = torch.zeros(self.opt.val_batch_size, 1).to(self.opt.device)
        else:
            max_len = 250
            prediction = torch.zeros(self.opt.test_batch_size, 1).to(self.opt.device)

        predictions = []
        hidden_states = [None, None]
        for i in range(max_len):
            if(self.training):
                if random.random() > self.opt.teacher_forcing_ratio:
                    char_embed = self.embedding(prediction.argmax(dim = 1))
                else:
                    char_embed = embeddings[:,i,:]
            else:
                noise = self.gumbel.sample(prediction.size()).to(self.opt.device)
                noise = noise.squeeze(2)
                prediction += noise
                char_embed = self.embedding(prediction.argmax(dim = 1))

            context, atten = self.attention(key, char_embed, values, lens)
            inp = torch.cat([char_embed, context], dim=1)
            hidden_states[0] = self.lstm1(inp, hidden_states[0])

            inp_2 = hidden_states[0][0]
            hidden_states[1] = self.lstm2(inp_2,hidden_states[1])

            output = hidden_states[1][0]
            prediction = self.character_prob(torch.cat([output, context], dim=1))
            predictions.append(prediction.unsqueeze(1))

        return torch.cat(predictions, dim=1)




