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