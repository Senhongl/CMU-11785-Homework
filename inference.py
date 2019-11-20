import time
import torch
import numpy as np
import pandas as pd
from torch.optim import Adam
from options import *
from Encoder import *
from Decoder import *
from util import *


def inference(opt, encoder, decoder, test_loader):
    encoder.eval()
    decoder.eval()

    result = []
    for batch_idx, (utterances, u_lens) in enumerate(test_loader):
        utterances = utterances.permute(1, 0, 2)
        utterances = utterances.to(opt.device)
        u_lens = u_lens.to(opt.device)

        keys, values, out_lens = encoder(utterances, u_lens)

        keys = keys.permute(1, 0, 2)
        values = values.permute(1, 0, 2)
        predict_labels = decoder(keys, values, lens = out_lens, mode = 'test').permute(0, 2, 1)
        
        result.append(predict_labels)

    return result

if __name__ == '__main__':
    opt = BaseOptions().parser.parse_args()
    speech_test = np.load(opt.dataroot + 'test_new.npy', allow_pickle=True, encoding='bytes')
    
    encoder = Encoder(opt)
    decoder = Decoder(opt)
    encoder.load_state_dict(torch.load('./' + opt.model_name + '/encoder_2.pt'))
    decoder.load_state_dict(torch.load('./' + opt.model_name + '/decoder_2.pt'))
    encoder.to(opt.device)
    decoder.to(opt.device)

    test_data = TestDataset(speech_test)
    test_loader_args = dict(shuffle=False, batch_size = opt.test_batch_size, pin_memory=True, collate_fn = collate_fn_test) 
    test_loader = Data.DataLoader(test_data, **test_loader_args)

    result = inference(opt, encoder, decoder, test_loader)

    result = transform_index_to_letter(result)

    dataframe = pd.DataFrame({'Id':[i for i in range(len(result))],'Predicted':result})

    dataframe.to_csv("submission.csv", index=False)
