import time
import torch
import numpy as np
from torch.optim import Adam
from options.TrainOptions import *
from Encoder import *
from Decoder import *
from util import *

def training(opt, encoder, decoder, train_loader, val_loader):
	encoder.to(opt.device)
	decoder.to(opt.device)
	encoder.train()
	decoder.train()

	for epoch in range(opt.n_epoch):
		avg_loss = 0
		start = time.time()
		for batch_idx, (utterances, labels, u_lens, l_lens) in enumerate(train_loader):
			
			utterances = utterances.to(opt.device)
			labels = labels.to(opt.device)
			u_lens = u_lens.to(opt.device)
			l_lens = l_lens.to(opt.device)

			encoder.zero_grad()
			decoder.zero_grad()
			optimizer.zero_grad()

			keys, values, out_lens = encoder(utterances, u_lens)
			predict_labels = decoder(keys, values, labels)
			
			loss = criterion(predict_labels, labels)
			mask = torch.arange(labels.size(1)).unsqueeze(0) >= l_lens.unsqueeze(1)
			
			loss.backward()
		# 	# `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
		# 	nn.utils.clip_grad_norm_(model.parameters(), 0.25)
		# 	avg_loss += loss.item()
		# 	if batch_idx % 10 == 9:
		# 		print('batch', batch_idx + 1, 'Loss', avg_loss / 10, 'Running time', time.time() - start)
		# 		avg_loss = 0
			
		# 	optimizer.step()

		# torch.save(model.state_dict(), './model_{}.pt'.format(epoch))
		# validation(model, val_loader)

if __name__ == '__main__':
	opt = TrainOptions().parser.parse_args()
	speech_train = np.load(opt.dataroot + 'train_new.npy', allow_pickle=True, encoding='bytes')
	speech_valid = np.load(opt.dataroot + 'dev_new.npy', allow_pickle=True, encoding='bytes')
	# speech_test = np.load(opt.dataroot + 'test.npy', allow_pickle=True, encoding='bytes')

	transcript_train = np.load(opt.dataroot + 'train_transcripts.npy', allow_pickle=True,encoding='bytes')
	transcript_valid = np.load(opt.dataroot + 'dev_transcripts.npy', allow_pickle=True,encoding='bytes')
	print("Data Loading Sucessful.....")
	encoder = Encoder(opt)
	decoder = Decoder(opt)
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

