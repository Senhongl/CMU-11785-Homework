import time
import torch
import numpy as np
from torch.optim import Adam
from options import *
from Encoder import *
from Decoder import *
from util import *
from Levenshtein import distance as levenshtein_distance

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
            optimizer_encoder.zero_grad()
            optimizer_decoder.zero_grad()

            outputs, hidden = encoder(utterances, u_lens)
            hidden = (hidden[0].permute(1, 0, 2), hidden[1].permute(1, 0, 2))
            hidden = (hidden[0].reshape(hidden[0].size(0), -1), hidden[1].reshape(hidden[1].size(0), -1))

            predict_labels = decoder(hidden, labels)
            predict_labels = predict_labels.permute(0, 2, 1)
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
                    opt_file.write('batch = {}, Perplexity = {}, Running time = {}'.format(batch_idx + 1, avg_loss / opt.display_freq, time.time() - start))
                    opt_file.write('\n')
                avg_loss = 0

            if batch_idx % (opt.display_freq * 10) == (opt.display_freq * 10 - 1):
                tmp_pred = transform_index_to_letter(predict_labels.unsqueeze(0))[0]
                tmp_true = np.array(letter_list)[labels[0].detach().cpu().numpy()]
                file_name_pred_train = os.path.join('./' + opt.model_name, '{}_pred_train.txt'.format(opt.model_name))
                with open(file_name_pred_train, 'a') as opt_file:
                    opt_file.write('prediction = {} Ground truth = {}'.format(tmp_pred, tmp_true))
                    opt_file.write('\n')
                
            optimizer_encoder.step()
            optimizer_decoder.step()

        validation(opt, encoder, decoder, val_loader)

def validation(opt, encoder, decoder, val_loader):
    encoder.eval()
    decoder.eval()

    start = time.time()
    running_loss = 0
    total_seq = 0
    score = 0
    for batch_idx, (utterances, labels, u_lens, l_lens) in enumerate(val_loader):
        utterances = utterances.permute(1, 0, 2)
        utterances = utterances.to(opt.device)
        labels = labels.to(opt.device)
        u_lens = u_lens.to(opt.device)
        l_lens = l_lens.to(opt.device)

        outputs, hidden = encoder(utterances, u_lens)
        hidden = (hidden[0].permute(1, 0, 2), hidden[1].permute(1, 0, 2))
        hidden = (hidden[0].reshape(hidden[0].size(0), -1), hidden[1].reshape(hidden[1].size(0), -1))

        predict_labels = decoder(hidden, labels)
        predict_labels = predict_labels.permute(0, 2, 1)
        loss = criterion(predict_labels, labels)
        mask = torch.arange(labels.size(1)).unsqueeze(0).to(opt.device) >= l_lens.unsqueeze(1)
        loss.masked_fill_(mask, 0.0)
        loss = loss.sum() / l_lens.sum()
        running_loss += torch.exp(loss).item()

        predict_labels = predict_labels.permute(0, 2, 1)
        for i in range(len(labels)):
            true_sentence = ''
            for j in range(l_lens[i] - 1):
                true_sentence += letter_list[labels[i][j]]

            predict_sentence = ''
            for j in range(len(predict_labels[i])):
                if predict_labels[i][j].argmax() == 33:
                    break
                predict_sentence += letter_list[predict_labels[i][j].argmax()]

            score += levenshtein_distance(true_sentence, predict_sentence)
            total_seq += 1

    print(score / total_seq)

    file_name = os.path.join('./' + opt.model_name, '{}.txt'.format(opt.model_name))
    with open(file_name, 'a') as opt_file:
        opt_file.write('='*16)
        opt_file.write('\n')
        opt_file.write('Perplexity = {}, Running time = {}'.format(running_loss / batch_idx, time.time() - start))
        opt_file.write('\n')

    predict_labels = predict_labels.permute(0, 2, 1)
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
    opt.model_name = 'base_model'
    options.printer(opt)
    speech_train = np.load(opt.dataroot + 'train_new.npy', allow_pickle=True, encoding='bytes')
    speech_valid = np.load(opt.dataroot + 'dev_new.npy', allow_pickle=True, encoding='bytes')

    transcript_train = np.load(opt.dataroot + 'train_transcripts.npy', allow_pickle=True,encoding='bytes')
    transcript_valid = np.load(opt.dataroot + 'dev_transcripts.npy', allow_pickle=True,encoding='bytes')
    print("Data Loading Sucessful.....")
    encoder = SimpleEncoder(opt)
    decoder = SimpleDecoder(opt)
    encoder.to(opt.device)
    decoder.to(opt.device)
    print(decoder)
    optimizer_encoder = Adam(encoder.parameters(), opt.lr, weight_decay = 1e-6)
    optimizer_decoder = Adam(decoder.parameters(), opt.lr, weight_decay = 1e-6)
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