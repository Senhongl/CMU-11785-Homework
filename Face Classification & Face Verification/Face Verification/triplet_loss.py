import torch
from torch import nn

class TripletLoss(nn.Module):
    def __init__(self, margin = 1):
        super(TripletLoss, self).__init__()
        self.margin = margin 

    def forward(self, inputs, labels):

        batch_size = inputs.shape[0]
        start_idx = 0
        mask = labels.expand(batch_size, batch_size).eq(labels.expand(batch_size, batch_size).t()) 
        loss = 0
        for idx in range(batch_size):
            anchor = inputs[idx]
            positive = inputs[mask[idx]]
            negative = inputs[mask[idx] == 0]
            ap = torch.norm((positive - anchor), p = 2, dim = 1)
            an = torch.norm((negative - anchor), p = 2, dim = 1)
            tmp = ap.mean() - an.min() + self.margin

            if tmp > 0:
                loss += tmp
            
        return loss




