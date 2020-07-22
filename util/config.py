from easydict import EasyDict
import torch

config = EasyDict()

# dataloader jobs number
config.num_workers = 4

# batch_size
config.batch_size = 4

# training epoch number
config.max_epoch = 200

config.start_epoch = 0

# learning rate, default: 1e-4

config.lr = 0.00001

# using GPU
config.cuda = False

config.n_disk = 30   # Default: 15

config.output_dir = 'output'

config.input_size = 512

# max polygon per image
config.max_annotation = 200

# max point per polygon, default: 20
config.max_points = 40

# use hard examples (annotated as '#')
config.use_hard = True

# demo tr threshold: bigger segmentated region if lower
config.tr_thresh = 0.1

# demo tcl threshold
config.tcl_thresh = 0.1

# expand ratio in post processing
config.post_process_expand = 0.2

# merge joined text instance when predicting
config.post_process_merge = False

def update_config(config, extra_config):

    for k, v in vars(extra_config).items():
        config[k] = v
    config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def print_config(config):

    print('==========Options============')
    for k, v in config.items():
        print('{}: {}'.format(k, v))
    print('=============End=============')
