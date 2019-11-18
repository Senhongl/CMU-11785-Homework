import argparse
import os
import torch

class BaseOptions:
    """This class defines options used during both training and test time.
    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self):
        self.initialize()

    def initialize(self):
        """Define the common options that are used in both training and test."""

        parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)
        # basic parameters
        parser.add_argument('--cuda', type = str, default = 'cuda', help = 'if use gpu, then it should be cuda, else cpu')
        parser.add_argument('--model_name', type = str, default = 'LAS_latest', help = 'the default filename to save log file and model would be LAS_latest')
        # model parameters
        parser.add_argument('--input_dim', type=int, default = 40, help='Actually for this dataset, it will only be 40, which corresponding to the input feature dimension')
        parser.add_argument('--encoder_hidden_dim', type = int, default = 256, help='the hidden dimension of enocder, according to the paper and writeup, 256 is large enough')
        parser.add_argument('--encoder_num_of_layers', type = int, default = 4, help = 'the number of hidden layers in the encoder, according to the paper and writeup, it should be 3')
        parser.add_argument('--encoder_is_bidirectional', type = bool, default = True, help = 'if we are using bidirectional for encoder, default should be true')
        parser.add_argument('--decoder_hidden_dim', type = int, default = 512, help = 'the number of hidden size of decoder, according to the paper and writeup, it should be 512')
        parser.add_argument('--embedding_size', type = int, default = 256, help = 'The embedding size we would use for decoder, according to the paper and the writeup, it should be 256')
        parser.add_argument('--decoder_num_layers', type = int, default = 2, help = 'The number of hidden layers we use for decoder, default should be 2')

        self.parser = parser
        return parser

    # def gather_options(self):
    #     """Initialize our parser with basic options(only once).
    #     """
    #     if not self.initialized:  # check if it has been initialized
            

    #     # save and return the parser
    #     self.parser = parser
    #     return parser.parse_args()

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
        os.mkdir(opt.model_name)
        file_name = os.path.join('./' + opt.model_name, '{}.txt'.format(opt.model_name))
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    # def parse(self):
    #     """Parse our options, create checkpoints directory suffix, and set up gpu device."""
    #     opt = self.gather_options()
    #     opt.isTrain = self.isTrain   # train or test

    #     # process opt.suffix
    #     if opt.suffix:
    #         suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
    #         opt.name = opt.name + suffix

    #     self.print_options(opt)

    #     # set gpu ids
    #     str_ids = opt.gpu_ids.split(',')
    #     opt.gpu_ids = []
    #     for str_id in str_ids:
    #         id = int(str_id)
    #         if id >= 0:
    #             opt.gpu_ids.append(id)
    #     if len(opt.gpu_ids) > 0:
    #         torch.cuda.set_device(opt.gpu_ids[0])

    #     self.opt = opt
    #     return self.opt


