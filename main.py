import os
from wsj_loader import WSJ
from dataset import * 
from torch.utils import data as Data

# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# os.environ['WSJ_PATH'] = './data'
# loader = WSJ()
# trainX, trainY = loader.train
# devX, devY = loader.dev

# train_data = MyDataset(trainX, trainY)
# dev_data = MyDataset(devX, devY)

# train_loader_args = dict(batch_size = 64, pin_memory=True, shuffle = True, collate_fn = collate_fn) 
# train_loader = Data.DataLoader(train_data, **train_loader_args)
# dev_loader_args = dict(shuffle=False, batch_size = 256, pin_memory=True, collate_fn = collate_fn) 
# dev_loader = Data.DataLoader(dev_data, **dev_loader_args)

letter_list = ['<sos>', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q',\
             'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '-', "'", '.', '_', '+', ' ','<eos>']

