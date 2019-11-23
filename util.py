import numpy as np
from torch.utils import data as Data
from torch.autograd import Variable
import torch
import matplotlib.pyplot as plt
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
        labels.append(torch.tensor(batch[batch_idx][1][1:]))
        l_lens.append(len(batch[batch_idx][1]) - 1)

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

def collect_word(transcript_train, transcript_val):
    word_dict = {}
    word_list = []
    word_to_index_train = []
    word_to_index_val = []
    word_dict['<sos>'] = 0
    word_dict['<eos>'] = 1
    word_list.append('<sos>')
    word_list.append('<eos>')
    count = 2
    for utterance in transcript_train:
        tmp_list = []
        tmp_list.append(0)
        for byte in utterance:
            word = byte.decode('utf8')
            if word not in word_dict:
                word_dict[word] = count
                word_list.append(word)
                tmp_list.append(count)
                count += 1
            else:
                tmp_list.append(word_dict[word])

        tmp_list.append(1)
        word_to_index_train.append(tmp_list)

    for utterance in transcript_val:
        tmp_list = []
        tmp_list.append(0)
        for byte in utterance:
            word = byte.decode('utf8')
            if word not in word_dict:
                word_dict[word] = count
                word_list.append(word)
                tmp_list.append(count)
                count += 1
            else:
                tmp_list.append(word_dict[word])

        tmp_list.append(1)
        word_to_index_val.append(tmp_list)

    return word_dict, np.array(word_list), word_to_index_train, word_to_index_val

        

letter_dict = {'<sos>': 0, 'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8, 'I': 9, 'J': 10, 'K': 11, 'L': 12, 'M': 13, 'N': 14, 'O': 15, 'P': 16, 'Q': 17,\
'R': 18, 'S': 19, 'T': 20, 'U': 21, 'V': 22, 'W': 23, 'X': 24, 'Y': 25, 'Z': 26, '-': 27, "'": 28, '.': 29, '_': 30, '+': 31, ' ': 32, '<eos>': 33}

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
                letters.append(letter_dict[character])

            letters.append(letter_dict[' '])

        letters.append(letter_dict['<eos>'])
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

def visualize(opt, weight):
    '''
    visualize the attention weight during training
    :param weight : (output_sequence_len, input_sequence_len)
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # ax.set_yticks(range(weight.shape[0]))
    # ax.set_xticks(range(weight.shape[1]))
    # ax.set_xticklabels(xLabel)
    im = ax.imshow(weight, cmap=plt.cm.hot_r)
    plt.colorbar(im)
    plt.savefig('./' + opt.model_name + '/attention_weight.png')

