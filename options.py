import argparse
import os
import torch

class BaseOptions:
    """This class defines options used during both training and test time.
    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self):
        opt = self.initialize()
        
    def initialize(self):
        """Define the common options that are used in both training and test."""

        parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)
        # basic parameters
        parser.add_argument('--dataroot', type = str, default = './data/', help = 'the data root used for loading data')
        parser.add_argument('--device', type = str, default = 'cuda', help = 'if use gpu, then it should be cuda, else cpu')
        parser.add_argument('--model_name', type = str, default = 'LAS_latest', help = 'the default filename to save log file and model would be LAS_latest')
        # model parameters
        parser.add_argument('--input_dim', type=int, default = 40, help='Actually for this dataset, it will only be 40, which corresponding to the input feature dimension')
        parser.add_argument('--encoder_hidden_dim', type = int, default = 256, help='the hidden dimension of enocder, according to the paper and writeup, 256 is large enough')
        parser.add_argument('--encoder_num_layers', type = int, default = 4, help = 'the number of hidden layers in the encoder, according to the paper and writeup, it should be 3')
        parser.add_argument('--is_bidirectional', type = bool, default = True, help = 'if we are using bidirectional for encoder, default should be true')
        parser.add_argument('--vocab_size', type = int, default = 34, help = 'If the model is character-based model, then the vocab_size would be 34')
        parser.add_argument('--decoder_hidden_dim', type = int, default = 512, help = 'the number of hidden size of decoder, according to the paper and writeup, it should be 512')
        parser.add_argument('--embedding_size', type = int, default = 256, help = 'The embedding size we would use for decoder, according to the paper and the writeup, it should be 256')
        parser.add_argument('--decoder_num_layers', type = int, default = 2, help = 'The number of hidden layers we use for decoder, default should be 2')
        parser.add_argument('--value_size', type = int, default = 128, help = 'The value size of values')
        parser.add_argument('--key_size', type = int, default = 128, help = 'The key_size of keys')
        parser.add_argument('--tao', type = float, default = 0.1, help = 'tao for gumbel trick')
        parser.add_argument('--dropout', type = float, default = 0.5, help = 'locked dropout rate')

        # network displaying and saving parameters 
        parser.add_argument('--save_latest_freq', type=int, default = 3, help='frequency of saving the latest results')
        parser.add_argument('--display_freq', type = int, default = 10, help = 'frequency of showing training results on screen')
        # training parameters
        parser.add_argument('--n_epoch', type = int, default = 10, help = 'The number of epoch runs for training')
        parser.add_argument('--lr', type = float, default = 0.001, help = 'initial learning rate for adam')
        parser.add_argument('--train_batch_size', type = int, default = 64, help = 'The batch_size used for training data loader')
        parser.add_argument('--teacher_forcing_ratio', type = float, default = 0.9, help = 'teacher forces ration used during training')
        # validation parameters
        parser.add_argument('--val_batch_size', type = int, default = 256, help = 'The batch_size used for validation data loader')
        # validation parameters
        parser.add_argument('--test_batch_size', type = int, default = 256, help = 'The batch_size used for test data loader')

        self.parser = parser
        return self.parser.parse_args(args = [])
        

    def printer(self, opt):
        """Print and save options
        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        if not os.path.isdir(opt.model_name):
            os.mkdir(opt.model_name)
        file_name = os.path.join('./' + opt.model_name, '{}.txt'.format(opt.model_name))
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

