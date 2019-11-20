import numpy as np
from torch.utils import data as Data
from torch.autograd import Variable
import torch
from torch.nn.utils.rnn import *

class MyDataset(Data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self,index):
        utterance = self.x[index]
        label = self.y[index]


        return (utterance, label)

class TestDataset(Data.Dataset):
    def __init__(self, x):
        self.x = x

    def __len__(self):
        return len(self.x)

    def __getitem__(self,index):
        utterance = self.x[index]

        return utterance

def collate_fn(batch):
    utterances = []
    u_lens = []
    labels = []
    l_lens = []
    for batch_idx in range(len(batch)):
        utterances.append(torch.tensor(batch[batch_idx][0]))
        u_lens.append(len(batch[batch_idx][0]))
        labels.append(torch.tensor(batch[batch_idx][1]))
        l_lens.append(len(batch[batch_idx][1]))

    utterances = pad_sequence(utterances, batch_first = True)
    labels = pad_sequence(labels, batch_first = True)
    u_lens = torch.tensor(u_lens)
    l_lens = torch.tensor(l_lens)
    return utterances, labels, u_lens, l_lens

def collate_fn_test(batch):
    utterances = []
    u_lens = []
    for batch_idx in range(len(batch)):
        utterances.append(torch.tensor(batch[batch_idx]))
        u_lens.append(len(batch[batch_idx]))

    utterances = pad_sequence(utterances, batch_first = True)
    u_lens = torch.tensor(u_lens)

    return utterances, u_lens

letter_list = ['<sos>', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q',\
'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '-', "'", '.', '_', '+', ' ','<eos>']

def transform_letter_to_index(transcript):
    '''
    :param transcript : Transcripts are the text input
    :return letter_to_index_list : Returns a list for all the transcript sentence to index
    '''
    letter_to_index_list = []
    for utterance in transcript:
        letters = []
        letters.append(letter_list.index('<sos>'))
        for byte in utterance:
            string = byte.decode('utf8')

            for character in string:
                letters.append(letter_list.index(character))

            letters.append(letter_list.index(' '))

        letters.append(letter_list.index('<eos>'))
        letter_to_index_list.append(letters)

    return letter_to_index_list

def transform_index_to_letter(transcript):
    '''
    :param transcript : Transcripts are the prediction input
    :return index_to_letter_list : Returns a list for all the index to sentence
    '''
    index_to_letter_list = []
    for batch_utterances in transcript:
        letters = ''
        for utterance in batch_utterances:
            utterance = utterance.argmax(dim = 0)
            for idx in utterance:
                letters += letter_list[idx]

            index_to_letter_list.append(letters)

    return index_to_letter_list