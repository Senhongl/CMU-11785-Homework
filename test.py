from Levenshtein import distance as levenshtein_distance
import time
import torch
import numpy as np
from torch.optim import Adam
from options import *
from Encoder import *
from Decoder import *
from util import *
from torch.utils.tensorboard import SummaryWriter



if __name__ == '__main__':
    options = BaseOptions()
    opt = options.parser.parse_args()
    # options.printer(opt)
    speech_train = np.load(opt.dataroot + 'train_new.npy', allow_pickle=True, encoding='bytes')
    speech_valid = np.load(opt.dataroot + 'dev_new.npy', allow_pickle=True, encoding='bytes')

    transcript_train = np.load(opt.dataroot + 'train_transcripts.npy', allow_pickle=True,encoding='bytes')
    transcript_valid = np.load(opt.dataroot + 'dev_transcripts.npy', allow_pickle=True,encoding='bytes')
    print("Data Loading Sucessful.....")
    # encoder = Encoder(opt)
    # decoder = Decoder(opt)
    # encoder.load_state_dict(torch.load('./LAS_latest/encoder_latest.pt'))
    # decoder.load_state_dict(torch.load('./LAS_latest/decoder_latest.pt'))
    # encoder.to(opt.device)
    # decoder.to(opt.device)
    # print(decoder)
    # optimizer_encoder = Adam(encoder.parameters(), opt.lr, weight_decay = 1e-6)
    # optimizer_decoder = Adam(decoder.parameters(), opt.lr, weight_decay = 1e-6)
    # criterion = nn.CrossEntropyLoss(reduction = 'none')
    # criterion.to(opt.device)

    # transcript_train = transform_letter_to_index(transcript_train)
    # transcript_valid = transform_letter_to_index(transcript_valid)

    print("Transfer the transcript from letters to index sucessfully.....")
    train_data = MyDataset(speech_train, transcript_train)
    dev_data = MyDataset(speech_valid, transcript_valid)

    train_loader_args = dict(batch_size = 1, pin_memory=True, shuffle = True, collate_fn = collate_fn) 
    train_loader = Data.DataLoader(train_data, **train_loader_args)
    dev_loader_args = dict(shuffle=False, batch_size = opt.val_batch_size, pin_memory=True, collate_fn = collate_fn) 
    dev_loader = Data.DataLoader(dev_data, **dev_loader_args)
    

    for batch_idx, (utterances, labels, u_lens, l_lens) in enumerate(train_loader):
        print(labels, l_lens)


