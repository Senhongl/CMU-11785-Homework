from .base_options import *

class TrainOptions(BaseOptions):
	"""This class includes training options.
	It also includes shared options defined in BaseOptions.
	"""
	def __init__(self):
		self.initialize()

	def initialize(self):
		self.parser = BaseOptions().parser
		# network displaying and saving parameters 
		self.parser.add_argument('--save_latest_freq', type=int, default = 1, help='frequency of saving the latest results')
		self.parser.add_argument('--display_freq', type = int, default = 50, help = 'frequency of showing training results on screen')
		# training parameters
		self.parser.add_argument('--n_epoch', type = int, default = 10, help = 'The number of epoch runs for training')
		self.parser.add_argument('--lr', type = float, default = 0.0001, help = 'initial learning rate for adam')
		self.parser.add_argument('--train_batch_size', type = int, default = 64, help = 'The batch_size used for training data loader')
		self.parser.add_argument('--teacher_forcing_ratio', type = int, default = 0.9, help = 'teacher forces ration used during training')
		# validation parameters
		self.parser.add_argument('--val_batch_size', type = int, default = 64, help = 'The batch_size used for validation data loader')


