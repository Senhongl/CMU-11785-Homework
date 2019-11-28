from Atten import *
from torch import nn
import numpy as np
import torch
import torch.nn.functional as F
import random
from torch.distributions.gumbel import Gumbel
from torch.autograd import Variable
from dropout import *

class Decoder(nn.Module):
    def __init__(self, opt):
        super(Decoder, self).__init__()
        self.opt = opt
        self.embedding = nn.Embedding(opt.vocab_size, opt.embedding_size)
        self.attention = Attention(opt)
        self.gumbel = Gumbel(torch.tensor([0.0]), torch.tensor([1.0]))
        self.lstm1 = nn.LSTMCell(input_size = opt.embedding_size + opt.value_size, hidden_size = opt.decoder_hidden_dim)
        self.drop1 = LockedDropout(dropout = opt.dropout)
        self.lstm2 = nn.LSTMCell(input_size = opt.decoder_hidden_dim, hidden_size = opt.key_size)
        self.drop2 = LockedDropout(dropout = opt.dropout)
        # self.linear = nn.Linear(opt.embedding_size, opt.key_size)
        self.key_network = nn.Linear(opt.encoder_hidden_dim * 2, opt.value_size)
        self.value_network = nn.Linear(opt.encoder_hidden_dim * 2, opt.key_size)
        self.query_network = nn.Linear(opt.embedding_size, opt.key_size)
        
        self.fc = nn.Linear(opt.key_size * 2, opt.embedding_size)
        self.tanh = nn.Hardtanh(inplace = True)
        self.character_prob = nn.Linear(opt.embedding_size, opt.vocab_size)
        self.character_prob.weight = self.embedding.weight

        

    def forward(self, encoder_out, text = None, lens = None, hidden = None, mode = 'train'):
        '''
        :param key :(N, T, key_size) Output of the Encoder Key projection layer
        :param values: (N, T, value_size) Output of the Encoder Value projection layer
        :param text: (N, text_len) Batch input of text with text_length
        :param lens: (N, ) Batch input of sequence length
        :return predictions: Returns the character perdiction probability 
        '''

        if mode == 'train' or mode == 'pre_training' or mode == 'val':
            max_len =  text.shape[1]
            # the shape of embeddings would be (N, text_len, embed_size)
            embeddings = self.embedding(text)
            prediction = torch.zeros(text.shape[0], 1).to(self.opt.device)
        else:
            max_len = 250
            prediction = torch.zeros(encoder_out.size(0), 1).to(self.opt.device)

        predictions = []
        hidden_states = [hidden, None]

        key = self.key_network(encoder_out)
        value = self.value_network(encoder_out)

        attentions = []
        for i in range(max_len):
            if(self.training):
                if random.random() > self.opt.teacher_forcing_ratio:
                    teacher_forcing = False
                else:
                    teacher_forcing = True

                if not teacher_forcing:
                    # noise = self.gumbel.sample(prediction.size()).to(self.opt.device)
                    # noise = Variable(noise.squeeze(2), requires_grad = True)
                    # prediction = prediction + noise
                    # prediction = F.softmax(prediction / self.opt.tao, dim = 1)
                    char_embed = self.embedding(prediction.argmax(dim = 1))
                else:
                    if i == 0:
                        char_embed = self.embedding(prediction.argmax(dim = 1))
                    else:
                        char_embed = embeddings[:, i - 1, :]
            else:
                # noise = self.gumbel.sample(prediction.size()).to(self.opt.device)
                # noise = Variable(noise.squeeze(2), requires_grad = True)
                # prediction = torch.log(prediction) + noise
                # if i == 0:
                #     char_embed = self.embedding(prediction.argmax(dim = 1))
                # else:
                #     char_embed = embeddings[:, i - 1, :]
                if i == 0:
                    char_embed = self.embedding(prediction.argmax(dim = 1))
                else:
                    char_embed = embeddings[:, i - 1, :]
                # char_embed = self.embedding(prediction.argmax(dim = 1))

            # char_embed = self.dropout(char_embed.unsqueeze(1)).squeeze(1)
            # convert query from (N, E) -> (N, key_size)
            query = self.query_network(char_embed)

            context, atten = self.attention(key, value, query, lens)

            inp = torch.cat([char_embed, context], dim=1)
            # inp = context
            inp = self.drop1(inp.unsqueeze(1)).squeeze(1)
            hidden_states[0] = self.lstm1(inp, hidden_states[0])

            # after_drop = self.dropout(hidden_states[0][0].unsqueeze(1)).squeeze(1)
            # hidden_states[0] = (self.dropout(hidden_states[0][0].unsqueeze(1)).squeeze(1), self.dropout(hidden_states[0][1].unsqueeze(1)).squeeze(1))
            inp_2 = hidden_states[0][0]
            inp_2 = self.drop2(inp_2.unsqueeze(1)).squeeze(1)
            hidden_states[1] = self.lstm2(inp_2,hidden_states[1])

            # after_drop = self.dropout(hidden_states[1][0].unsqueeze(1)).squeeze(1)
            # hidden_states[1] = (self.dropout(hidden_states[1][0].unsqueeze(1)).squeeze(1), self.dropout(hidden_states[1][1].unsqueeze(1)).squeeze(1))
            
            outputs = hidden_states[1][0]

            query = self.query_network(outputs)
            context, atten = self.attention(key, value, query, lens)

            # store attention weight to visualize it
            attentions.append(atten.detach().cpu().numpy())

            fc_outputs = self.fc(torch.cat([outputs, context], dim=1))
            fc_outputs = self.tanh(fc_outputs)
            prediction = self.character_prob(fc_outputs)

            predictions.append(prediction.unsqueeze(1))

        return torch.cat(predictions, dim = 1), np.array(attentions)[:, 0, :lens[0]]

    def Greedy(self, encoder_out, lens = None, hidden = None):
        max_len = 250
        prediction = torch.zeros(encoder_out.size(0), 1).to(self.opt.device)
        key = self.key_network(encoder_out)
        value = self.value_network(encoder_out)
        predictions = []
        hidden_states = [hidden, None]

        for idx in range(max_len):
            char_embed = self.embedding(prediction.argmax(dim = 1))
            
            query = self.query_network(char_embed)

            context, atten = self.attention(key, value, query, lens)

            inp = torch.cat([char_embed, context], dim=1)
            # inp = context
            # inp = self.drop1(inp.unsqueeze(1)).squeeze(1)
            hidden_states[0] = self.lstm1(inp, hidden_states[0])

            # after_drop = self.dropout(hidden_states[0][0].unsqueeze(1)).squeeze(1)
            # hidden_states[0] = (self.dropout(hidden_states[0][0].unsqueeze(1)).squeeze(1), self.dropout(hidden_states[0][1].unsqueeze(1)).squeeze(1))
            inp_2 = hidden_states[0][0]
            # inp_2 = self.drop2(inp_2.unsqueeze(1)).squeeze(1)
            hidden_states[1] = self.lstm2(inp_2,hidden_states[1])

            # after_drop = self.dropout(hidden_states[1][0].unsqueeze(1)).squeeze(1)
            # hidden_states[1] = (self.dropout(hidden_states[1][0].unsqueeze(1)).squeeze(1), self.dropout(hidden_states[1][1].unsqueeze(1)).squeeze(1))
            
            outputs = hidden_states[1][0]

            query = self.query_network(outputs)
            context, atten = self.attention(key, value, query, lens)

            fc_outputs = self.fc(torch.cat([outputs, context], dim=1))
            fc_outputs = self.tanh(fc_outputs)
            prediction = self.character_prob(fc_outputs)

            predictions.append(prediction.unsqueeze(1))

        return torch.cat(predictions, dim = 1)


    def BeamSearch(self, encoder_out, lens = None, hidden = None, beam_width = 10):
        max_len = 250

        key = self.key_network(encoder_out)
        value = self.value_network(encoder_out)
        predictions = []
        hidden_states = [hidden, None]
        Path = [{'hypothesis': torch.LongTensor([0]).to(self.opt.device), 'score': 0, 'hidden_states': [hidden, None], 'history_path': []}]
        complete_hypothesis = []

        for idx in range(max_len):
            tmp_path = []
            for path in Path:
                char_embed = self.embedding(path['hypothesis'])

                query = self.query_network(char_embed.squeeze()).unsqueeze(0)

                context, atten = self.attention(key, value, query, lens)

                inp = torch.cat([char_embed, context], dim=1)
                
                # extract hidden states from path
                hidden_states = path['hidden_states']

                hidden_states[0] = self.lstm1(inp, hidden_states[0])
            
                inp_2 = hidden_states[0][0]

                hidden_states[1] = self.lstm2(inp_2,hidden_states[1])

                outputs = hidden_states[1][0]

                query = self.query_network(outputs)
                context, atten = self.attention(key, value, query, lens)

                fc_outputs = self.fc(torch.cat([outputs, context], dim=1))
                fc_outputs = self.tanh(fc_outputs)
                character_prob_distrib = self.character_prob(fc_outputs)

                local_score, local_idx = F.log_softmax(character_prob_distrib, dim = 1).topk(beam_width)

                for tmp_beam_idx in range(beam_width):
                    tmp_dict = {}
                    tmp_dict['score'] = path['score'] + local_score[0][tmp_beam_idx]
                    tmp_dict['hypothesis'] = local_idx[:, tmp_beam_idx]
                    tmp_dict['history_path'] = path['history_path'] + [local_idx[:, tmp_beam_idx]]
                    tmp_dict['hidden_states'] = hidden_states

                    # if the idx is <eos> idx, get it to the complete hypothesis set
                    if local_idx[0][tmp_beam_idx] == 33:
                        complete_hypothesis.append(tmp_dict)
                    # else, store it and compare the score at the end
                    else:
                        tmp_path.append(tmp_dict)

            if len(complete_hypothesis) == beam_width:
                break
            else:
                Path = sorted(tmp_path, key = lambda p : p['score'], reverse = True)[:beam_width]

        best_one = sorted(complete_hypothesis, key = lambda p : p['score'] / len(p['history_path']), reverse = True)[0]

        return best_one['history_path'][:-1]



class SimpleDecoder(nn.Module):
    """Simple baseline model to check if otherthings works well
    """
    def __init__(self, opt):
        super(SimpleDecoder, self).__init__()
        self.opt = opt
        self.embed = nn.Embedding(opt.vocab_size, opt.embedding_size)
        self.lstm = nn.LSTMCell(input_size = opt.embedding_size, hidden_size = opt.decoder_hidden_dim)
        self.fc = nn.Linear(opt.decoder_hidden_dim, opt.vocab_size)

    def forward(self, hidden = None, text = None):

        max_len = text.shape[1]
        prediction = torch.zeros(text.size(0), 1).to(self.opt.device)
        predictions = []

        for i in range(max_len):
            
            char_embed = self.embed(prediction.argmax(dim = 1))

            hidden = self.lstm(char_embed, hidden)
            prediction = self.fc(hidden[0])
            predictions.append(prediction.unsqueeze(1))
        
        return torch.cat(predictions, dim=1)

