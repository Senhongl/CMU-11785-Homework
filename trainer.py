import time
import torch
import numpy as np
from torch.optim import Adam
from options import *
from Encoder import *
from Decoder import *
from util import *

def training(opt, encoder, decoder, train_loader, val_loader):
    encoder.train()
    decoder.train()

    for epoch in range(opt.n_epoch):
        avg_loss = 0
        start = time.time()
        for batch_idx, (utterances, labels, u_lens, l_lens) in enumerate(train_loader):
            utterances = utterances.permute(1, 0, 2)
            utterances = utterances.to(opt.device)
            labels = labels.to(opt.device)
            u_lens = u_lens.to(opt.device)
            l_lens = l_lens.to(opt.device)
            
            encoder.zero_grad()
            decoder.zero_grad()
            optimizer.zero_grad()

            keys, values, out_lens = encoder(utterances, u_lens)

            keys = keys.permute(1, 0, 2)
            values = values.permute(1, 0, 2)
            predict_labels = decoder(keys, values, labels, out_lens).permute(0, 2, 1)
            loss = criterion(predict_labels, labels)
            mask = torch.arange(labels.size(1)).unsqueeze(0).to(opt.device) >= l_lens.unsqueeze(1)
            loss.masked_fill_(mask, 0.0)
            loss = loss.sum() / l_lens.sum()
            avg_loss += torch.exp(loss).item()
            loss.backward()
			# `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            nn.utils.clip_grad_norm_(encoder.parameters(), 0.25)   
            nn.utils.clip_grad_norm_(decoder.parameters(), 0.25)   
            

            if batch_idx % opt.display_freq == (opt.display_freq - 1):
                file_name = os.path.join('./' + opt.model_name, '{}.txt'.format(opt.model_name))
                with open(file_name, 'a') as opt_file:
                    opt_file.write('batch = {}, Perplexity = {}, Running time = {}'.format(batch_idx + 1, avg_loss / 10, time.time() - start))
                    opt_file.write('\n')
                
                tmp_pred = transform_index_to_letter(predict_labels.unsqueeze(0))[0]
                tmp_true = np.array(letter_list)[labels[0].detach().cpu().numpy()]
                file_name_pred_train = os.path.join('./' + opt.model_name, '{}_pred_train.txt'.format(opt.model_name))
                with open(file_name_pred_train, 'a') as opt_file:
                    opt_file.write('true = {}, predict = {}'.format(tmp_pred, tmp_true))
                    opt_file.write('\n')
                avg_loss = 0
            optimizer.step()

        if epoch % opt.save_latest_freq == (opt.save_latest_freq - 1):
            torch.save(encoder.state_dict(), './{}/encoder_{}.pt'.format(opt.model_name, epoch))
            torch.save(decoder.state_dict(), './{}/decoder_{}.pt'.format(opt.model_name, epoch))
            torch.save(encoder.state_dict(), './{}/encoder_latest.pt'.format(opt.model_name, epoch))
            torch.save(decoder.state_dict(), './{}/decoder_latest.pt'.format(opt.model_name, epoch))
        validation(opt, encoder, decoder, val_loader)

def validation(opt, encoder, decoder, val_loader):
    encoder.eval()
    decoder.eval()

    start = time.time()
    running_loss = 0
    for batch_idx, (utterances, labels, u_lens, l_lens) in enumerate(val_loader):
        utterances = utterances.permute(1, 0, 2)
        utterances = utterances.to(opt.device)
        labels = labels.to(opt.device)
        u_lens = u_lens.to(opt.device)
        l_lens = l_lens.to(opt.device)

        keys, values, out_lens = encoder(utterances, u_lens)

        keys = keys.permute(1, 0, 2)
        values = values.permute(1, 0, 2)
        predict_labels = decoder(keys, values, labels, out_lens, mode = 'val').permute(0, 2, 1)
        loss = criterion(predict_labels, labels)
        mask = torch.arange(labels.size(1)).unsqueeze(0).to(opt.device) >= l_lens.unsqueeze(1)
        loss.masked_fill_(mask, 0.0)
        loss = loss.sum() / l_lens.sum()
        running_loss += torch.exp(loss).item()
    
    file_name = os.path.join('./' + opt.model_name, '{}.txt'.format(opt.model_name))
    with open(file_name, 'a') as opt_file:
        opt_file.write('='*16)
        opt_file.write('\n')
        opt_file.write('Perplexity = {}, Running time = {}'.format(running_loss / batch_idx, time.time() - start))
        opt_file.write('\n')

    tmp_pred = transform_index_to_letter(predict_labels.unsqueeze(0))[0]
    tmp_true = np.array(letter_list)[labels[0].detach().cpu().numpy()]
    file_name_pred_val = os.path.join('./' + opt.model_name, '{}_pred_val.txt'.format(opt.model_name))
    with open(file_name_pred_val, 'a') as opt_file:
        opt_file.write('true = {}, predict = {}'.format(tmp_pred, tmp_true))
        opt_file.write('\n')

    encoder.train()
    decoder.train()




if __name__ == '__main__':
    options = BaseOptions()
    opt = options.parser.parse_args()
    options.printer(opt)
    speech_train = np.load(opt.dataroot + 'train_new.npy', allow_pickle=True, encoding='bytes')
    speech_valid = np.load(opt.dataroot + 'dev_new.npy', allow_pickle=True, encoding='bytes')

    transcript_train = np.load(opt.dataroot + 'train_transcripts.npy', allow_pickle=True,encoding='bytes')
    transcript_valid = np.load(opt.dataroot + 'dev_transcripts.npy', allow_pickle=True,encoding='bytes')
    print("Data Loading Sucessful.....")
    encoder = Encoder(opt)
    decoder = Decoder(opt)
    encoder.to(opt.device)
    decoder.to(opt.device)
    optimizer = Adam(list(encoder.parameters()) + list(decoder.parameters()), opt.lr)
    criterion = nn.CrossEntropyLoss(reduction = 'none')
    criterion.to(opt.device)

    transcript_train = transform_letter_to_index(transcript_train)
    transcript_valid = transform_letter_to_index(transcript_valid)

    print("Transfer the transcript from letters to index sucessfully.....")
    train_data = MyDataset(speech_train, transcript_train)
    dev_data = MyDataset(speech_valid, transcript_valid)

    train_loader_args = dict(batch_size = opt.train_batch_size, pin_memory=True, shuffle = True, collate_fn = collate_fn) 
    train_loader = Data.DataLoader(train_data, **train_loader_args)
    dev_loader_args = dict(shuffle=False, batch_size = opt.val_batch_size, pin_memory=True, collate_fn = collate_fn) 
    dev_loader = Data.DataLoader(dev_data, **dev_loader_args)

    training(opt, encoder, decoder, train_loader, dev_loader)

