from torch.utils import data as Data
from torch.autograd import Variable
import torch
from torch.nn.utils.rnn import *


class MyDataset(Data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = (y + 1)
    
    def __len__(self):
        return len(self.x)

    def __getitem__(self,index):
        utterance = self.x[index]
        label = self.y[index]
            
        
        return (utterance, label)
    
def collate_fn(batch):
#     batch.sort(key = lambda x: len(x[0]), reverse = True)
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
#     print('collate_fn', u_lens, l_lens)
    return utterances, labels, u_lens, l_lens



