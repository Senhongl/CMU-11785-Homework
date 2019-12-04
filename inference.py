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

        outs, out_lens, hidden = encoder(utterances, u_lens)
        
        hidden = (hidden[0].permute(1, 0, 2), hidden[1].permute(1, 0, 2))
        hidden = (hidden[0].reshape(hidden[0].size(0), -1), hidden[1].reshape(hidden[1].size(0), -1))

        outs = outs.permute(1, 0, 2)

        predict_labels = decoder.BeamSearch(outs, lens = out_lens, hidden = hidden)

        tmp_res = ''
        for i in predict_labels:
            tmp_res += letter_list[i]

        print(batch_idx, tmp_res)
        result.append(tmp_res)

        del utterances
        del u_lens
        del outs
        del out_lens
        torch.cuda.empty_cache()

    return result


# def inference(opt, encoder, decoder, test_loader):
#     encoder.eval()
#     decoder.eval()

#     result = []
#     for batch_idx, (utterances, u_lens) in enumerate(test_loader):
#         utterances = utterances.permute(1, 0, 2)
#         utterances = utterances.to(opt.device)
#         u_lens = u_lens.to(opt.device)

#         outs, out_lens, hidden = encoder(utterances, u_lens)
        
#         hidden = (hidden[0].permute(1, 0, 2), hidden[1].permute(1, 0, 2))
#         hidden = (hidden[0].reshape(hidden[0].size(0), -1), hidden[1].reshape(hidden[1].size(0), -1))

#         outs = outs.permute(1, 0, 2)
#         predict_labels = decoder.Greedy(outs, lens = out_lens, hidden = hidden)
#         predict_labels = predict_labels.permute(0, 2, 1)

#         result.append(predict_labels)

#         del utterances
#         del u_lens
#         del outs
#         del out_lens
#         torch.cuda.empty_cache()

#     return result

if __name__ == '__main__':
    opt = BaseOptions().parser.parse_args()
    speech_test = np.load(opt.dataroot + 'test_new.npy', allow_pickle=True, encoding='bytes')
    
    encoder = Encoder(opt)
    decoder = Decoder(opt)
    encoder.load_state_dict(torch.load('./' + opt.model_name + '/encoder_latest.pt'))
    decoder.load_state_dict(torch.load('./' + opt.model_name + '/decoder_latest.pt'))
    encoder.to(opt.device)
    decoder.to(opt.device)

    criterion = nn.CrossEntropyLoss(reduction = 'none')
    criterion.to(opt.device)
    test_data = TestDataset(speech_test)
    test_loader_args = dict(shuffle=False, batch_size = 1, pin_memory=True, collate_fn = collate_fn_test) 
    test_loader = Data.DataLoader(test_data, **test_loader_args)

    result = inference(opt, encoder, decoder, test_loader)

    # tmp_result = transform_index_to_letter(tmp_result)

    # result = []
    # for utterance in tmp_result:
    #     for word_idx in range(len(utterance)):
    #         if utterance[word_idx] == '<':
    #             break
    #     result.append(utterance[:word_idx - 1])

    dataframe = pd.DataFrame({'Id':[i for i in range(len(result))],'Predicted':result})

    dataframe.to_csv("submission.csv", index=False)
