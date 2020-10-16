#!/usr/bin/env python
# coding: utf-8

# In[64]:


import os
from torch.utils import data as Data
import torch
from PIL import Image
import numpy as np
from torchvision import transforms
import torchvision
import time

data_transform = transforms.Compose([                           
    transforms.RandomHorizontalFlip(),                        
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[    
                             0.229, 0.224, 0.225])
         ])

data_transform_val = transforms.Compose([                                                   
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[    
                             0.229, 0.224, 0.225])
         ])

# In[3]:


train_data = torchvision.datasets.ImageFolder(root='./train_data/medium', transform = data_transform)
val_data = torchvision.datasets.ImageFolder(root='./validation_classification/medium', transform = data_transform_val)
train_loader_args = dict(batch_size= 128, pin_memory=True, shuffle = True, num_workers = 8) 
train_loader = Data.DataLoader(train_data, **train_loader_args)
val_loader_args = dict(shuffle=False, batch_size= 128, pin_memory=True, num_workers = 8) 
val_loader = Data.DataLoader(val_data, **val_loader_args)






import os
import numpy as np
from PIL import Image

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels, eps = 1e-05, momentum = 0.1, affine = True, track_running_stats = True),
            nn.Dropout(p = 0.1),
            nn.ReLU(inplace=True)
        )
        
class InvertedBottleneck(nn.Module):
    
    def __init__(self, in_channels, inter_channels):
        super(InvertedBottleneck, self).__init__()
        self.convbnrl1 = ConvBNReLU(in_channels, inter_channels, kernel_size = 1, stride = 1)
        self.convbnrl2 = ConvBNReLU(inter_channels, inter_channels, kernel_size = 3, stride = 1, groups = inter_channels)
        self.convbnrl3 = ConvBNReLU(inter_channels, inter_channels, kernel_size = 1, stride = 1)
        self.conv = nn.Conv2d(inter_channels, in_channels, kernel_size = 1, stride = 1, bias = False)
        self.bn = nn.BatchNorm2d(in_channels, eps = 1e-05, momentum = 0.1, affine = True, track_running_stats = True)

        
    def forward(self, x):
        out = self.convbnrl1(x)
        out = self.convbnrl2(out)
        out = self.convbnrl3(out)
        out = self.conv(out)
        out = self.bn(out)
        out += x
    
        return out

class Transit(nn.Module):
    
    def __init__(self, in_channels, out_channels, padding = 0):
        super(Transit, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = 1, padding = padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps = 1e-05, momentum = 0.1, affine = True, track_running_stats = True)
        self.rl = nn.ReLU(inplace=True)
        
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.rl(out)
        return out
    
class Block(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super(Block, self).__init__()
        self.IB1 = InvertedBottleneck(in_channels, 4 * in_channels)
        self.transit = Transit(in_channels, out_channels, padding = 1)
        self.IB2 = InvertedBottleneck(out_channels, 4 * out_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = 1, bias = False)
    
    def forward(self, x):
        out = self.IB1(x)
        out = self.transit(out)
        out = self.IB2(out)
        out += self.conv(x)

        
        return out


class Network(nn.Module):
    def __init__(self, num_feats, hidden_sizes, num_classes, feat_dim=10):
        super(Network, self).__init__()
        
        self.hidden_sizes = [num_feats] + hidden_sizes + [num_classes]
        
        self.layers = []
        
        self.layers.append(nn.Conv2d(num_feats, self.hidden_sizes[1], kernel_size = 3, stride = 2, bias = False))
        self.layers.append(nn.BatchNorm2d(self.hidden_sizes[1], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        self.layers.append(nn.ReLU(inplace = True))
        for idx in range(1, len(self.hidden_sizes) - 2):
            in_channels = self.hidden_sizes[idx]
            out_channels = self.hidden_sizes[idx + 1]
            
            self.layers.append(Block(in_channels, out_channels))
            
    
        self.layers = nn.Sequential(*self.layers)
        self.linear_label = nn.Linear(self.hidden_sizes[-2], self.hidden_sizes[-1])
        
        # For creating the embedding to be passed into the Center Loss criterion
        self.linear_closs = nn.Linear(self.hidden_sizes[-2], feat_dim, bias=False)
        self.relu_closs = nn.ReLU(inplace=True)
    
    def forward(self, x, evalMode=False):
        output = x
        output = self.layers(output)
            
        output = F.avg_pool2d(output, [output.size(2), output.size(3)], stride=1)
        output = output.reshape(output.shape[0], output.shape[1])
        
        label_output = self.linear_label(output)
        label_output = label_output/torch.norm(self.linear_label.weight, dim=1)
        
        # Create the feature embedding for the Center Loss
        closs_output = self.linear_closs(output)
        closs_output = self.relu_closs(closs_output)

        return closs_output, label_output

def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight.data)


# In[7]:


numEpochs = 100
num_feats = 3
closs_weight = 1
feat_dim = 2300

learningRate = 1e-2
weightDecay = 5e-5

hidden_sizes = [32, 64, 128, 256]
num_classes = 2300

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[11]:


from torch import optim

network = Network(num_feats, hidden_sizes, num_classes)
network.apply(init_weights)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(network.parameters(), lr = 0.0001)
cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")


# In[9]:


def train(model, data_loader, test_loader, task='Classification'):
    model.train()

    for epoch in range(numEpochs):
        start = time.time()
        avg_loss = 0.0
        for batch_num, (feats, labels) in enumerate(data_loader):
            feats, labels = feats.to(device), labels.to(device)
            
            optimizer.zero_grad()
            feature, outputs = model(feats)

            loss = criterion(outputs, labels.long())

            loss.backward()
            optimizer.step()
            
            avg_loss += loss.item()

            if batch_num % 50 == 49:
                print('Epoch: {}\tBatch: {}\tAvg-Loss: {:.4f}\tRunningtime: {:.4f}'.format(epoch+1, batch_num+1, avg_loss/50, time.time() - start))
                avg_loss = 0.0    
            
            torch.cuda.empty_cache()
            del feats
            del labels
            del loss
        
        if task == 'Classification':
            val_loss, val_acc = test_classify(model, test_loader)
#             train_loss, train_acc = test_classify(model, data_loader)
#             print('Train Loss: {:.4f}\tTrain Accuracy: {:.4f}\tVal Loss: {:.4f}\tVal Accuracy: {:.4f}'.
#                   format(train_loss, train_acc, val_loss, val_acc))
            print('Val Loss: {:.4f}\tVal Accuracy: {:.4f}'.
                  format(val_loss, val_acc))
        else:
            test_verify(model, test_loader)
        

        torch.save(model.state_dict(), './model_{}.pt'.format(epoch - 4))

def test_classify(model, test_loader):
    model.eval()
    test_loss = []
    accuracy = 0
    total = 0

    for batch_num, (feats, labels) in enumerate(test_loader):
        feats, labels = feats.to(device), labels.to(device)
        feature, outputs = model(feats)
        
        _, pred_labels = torch.max(F.softmax(outputs, dim=1), 1)
        pred_labels = pred_labels.view(-1)
        
        loss = criterion(outputs, labels.long())

        accuracy += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)
        test_loss.extend([loss.item()]*feats.size()[0])
        del feats
        del labels
    
    model.train()
    return np.mean(test_loss), accuracy/total


def test_verify(model, test_loader):
    raise NotImplementedError


network.train()
network.to(device)
train(network, train_loader, val_loader)

