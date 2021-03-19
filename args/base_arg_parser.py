"""
base_arg_parser.py
Base arguments for all scripts
"""

import argparse
import json
import numpy as np
import os
from os.path import dirname, join
import random
import subprocess
from datetime import datetime

import torch
import torch.backends.cudnn as cudnn


class BaseArgParser(object):
    """Base argument parser for args shared between test and train modes."""
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='disentanglement')

        self.parser.add_argument('--name', type=str, default='debug', help='Experiment name prefix.')
        self.parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducible outputs.')

        self.parser.add_argument('--model_generator', type=str, default='Pix2PixHDGenerator', help='Generator to use')
        self.parser.add_argument('--model_discriminator', type=str, default='Pix2PixHDDiscriminator', help='Discriminator to use')
        self.parser.add_argument('--csv_dir', type=str, default='/data2/braingan/debug', help='Directory where csv splits are.')
        self.parser.add_argument('--batch_size', type=int, default=4, help='Batch size.')
        self.parser.add_argument('--viz_batch_size', type=int, default=4, help='Visualization image batch size.')

        self.parser.add_argument('--dataset_name', type=str, default='dsprites', choices=('dpsrites', 'shapes3d', 'norb', 'cars3d', 'mpi3d', 'scream'), help='Dataset to use.')

        self.parser.add_argument('--gpu_ids', type=str, default='0,1', help='Comma-separated list of GPU IDs. Use -1 for CPU.')
        self.parser.add_argument('--num_workers', default=1, type=int, help='Number of threads for the DataLoader.')

        self.parser.add_argument('--init_method', type=str, default='kaiming', choices=('kaiming', 'normal', 'xavier'), help='Initialization method to use for conv kernels and linear weights.')

        self.parser.add_argument('--higher_metric_better', type=bool, default=False, help='For evaluation, higher the metric the better, else lower.')
        self.parser.add_argument('--input_scan', type=str, default='noncon', help='What input should the GAN use')
        self.parser.add_argument('--goal_scan', type=str, default='con', help='What is the GAN trying to create?')
        self.parser.add_argument('--save_dir', type=str, default='/data2/braingan/results', help='Directory for results, prefix.')
        self.parser.add_argument('--num_visuals', type=str, default=4, help='Number of visual examples to show per batch on Tensorboard.')

        self.parser.add_argument('--resample', type=str, default='32,400,400', help='Resampling for inputs')
        self.parser.add_argument('--slices', type=int, default=-1, help='Number of resampled slices are passed into the model as input')

    def parse_args(self):
        args = self.parser.parse_args()

        # Get version control hash for record-keeping
        args.commit_hash = subprocess.run(
            ['git', '-C', join('.', dirname(__file__)), 'rev-parse', 'HEAD'], stdout=subprocess.PIPE,
            universal_newlines=True
        ).stdout.strip()
        # This appends, if necessary, a message about there being uncommitted changes
        # (i.e. if there are uncommitted changes, you can't be sure exactly
        # what the code looks like, whereas if there are no uncommitted changes,
        # you know exactly what the code looked like).
        args.commit_hash += ' (with uncommitted changes)' if bool(subprocess.run(
            ['git', '-C', join('.', dirname(__file__)), 'status', '--porcelain'], stdout=subprocess.PIPE,
            universal_newlines=True
        ).stdout.strip()) else ''

        # Create save dir for run
        if not hasattr(args, 'continue_train') and args.continue_train:
            args.name = (f'{args.name}_{datetime.now().strftime("%b%d_%H%M%S")}'
                f'_{os.getlogin()}')
        save_dir = os.path.join(args.save_dir, f'{args.name}')
        print(args.name)
        print(save_dir)
        os.makedirs(save_dir, exist_ok=True)
        args.save_dir = save_dir

        # Create ckpt dir and viz dir
        args.ckpt_dir = os.path.join(args.save_dir, 'ckpts')
        os.makedirs(args.ckpt_dir, exist_ok=True)

        args.viz_dir = os.path.join(args.save_dir, 'viz')
        os.makedirs(args.viz_dir, exist_ok=True)

        # Set up available GPUs
        def args_to_list(csv, arg_type=int):
            """Convert comma-separated arguments to a list."""
            arg_vals = [arg_type(d) for d in str(csv).split(',')]
            return arg_vals

        args.resample = args_to_list(args.resample)
        args.slices = args.slices if args.slices != -1 else args.resample[0]
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
        args.gpu_ids = args_to_list(args.gpu_ids)

        if len(args.gpu_ids) > 0 and torch.cuda.is_available():
            # Set default GPU for `tensor.to('cuda')`
            # torch.cuda.set_device(args.gpu_ids[0])
            args.device = 'cuda'
        else:
            args.device = 'cpu'

        if hasattr(args, 'supervised_factors'):
            args.supervised_factors = args_to_list(args.supervised_factors)

        # Save args to a JSON file
        with open(os.path.join(save_dir, 'args.json'), 'w') as fh:
            json.dump(vars(args), fh, indent=4, sort_keys=True)
            fh.write('\n')

        return args
