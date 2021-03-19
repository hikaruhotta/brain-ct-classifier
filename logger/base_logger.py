import os
import torch

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from tensorboardX import SummaryWriter

import torch.nn.functional as F
import torchvision.utils as vutils


class BaseLogger(object):
    def __init__(self, args, dataset_len):

        def round_down(x, m):
            """Round x down to a multiple of m."""
            return int(m * round(float(x) / m))

        self.args = args
        self.batch_size = args.batch_size
        self.save_dir = args.save_dir
        self.num_visuals = args.num_visuals
        
        self.dataset_len = dataset_len

        # Tensorboard dir: /data2/braingan/results/tb/{name}/
        tb_dir = Path('/'.join(args.save_dir.split('/')[:-1])) / 'tb' / args.name
        os.makedirs(tb_dir, exist_ok=True)
        self.summary_writer = SummaryWriter(log_dir=tb_dir)

        # Path to log file
        self.log_filepath = os.path.join(self.save_dir, f'{args.name}.log')

        self.epoch = args.start_epoch
        self.iter = 0

        # Current iteration overall (i.e., total # of examples seen)
        self.global_step = round_down((self.epoch - 1) * dataset_len, args.batch_size)
        self.iter_start_time = None
        self.epoch_start_time = None

    def _log_text(self, text_dict):
        for k, v in text_dict.items():
            self.summary_writer.add_text(k, str(v), self.global_step)

    def _log_scalars(self, scalar_dict, print_to_stdout=True, unique_id=None):
        """Log all values in a dict as scalars to TensorBoard."""
        for k, v in scalar_dict.items():
            if print_to_stdout:
                self.write(f'[{k}: {v}]')
            k = k.replace('_', '/')  # Group in TensorBoard by split.
            if unique_id is not None:
                k = f'{k}/{unique_id}'
            self.summary_writer.add_scalar(k, v, self.global_step)

    def visualize(self, inputs, targets, generated, split, unique_id=None):
        """Visualize predictions and targets in TensorBoard.

        Args:
            inputs: Inputs to the model.
            targets: Target images for the inputs.
            generated: Generated images from the inputs.
            split: One of 'train' or 'test'.
            unique_id: A unique ID to append to every image title. Allows
              for displaying all visualizations separately on TensorBoard.

        Returns:
            Number of examples visualized to TensorBoard.
        """
        batch_size, _, num_slices, _, _ = inputs.shape

        for i in range(self.num_visuals):
            if i >= batch_size:
                # Exceeded number in batch
                break
            
            # Get the i-th volume in batch
            inputs_vol = torch.squeeze(inputs[i], 0)
            targets_vol = torch.squeeze(targets[i], 0)
            generated_vol = torch.squeeze(generated[i], 0)
            
            vis_list = []
            for s in range(num_slices):
                # One slice at a time
                stacked_slice = torch.stack([inputs_vol[s].unsqueeze(0),
                                             targets_vol[s].unsqueeze(0),
                                             generated_vol[s].unsqueeze(0)])
                
                # Concat along the width (dim=1)
                vis = torch.cat([ss.permute(1,2,0) for ss in stacked_slice], 1)
                vis_list.append(vis)

            # Concat along the height (dim=0)
            visuals = torch.cat([v.squeeze(-1) for v in vis_list])
            visuals_np = visuals.detach().to('cpu').numpy() * 255

            title = 'inputs-targets-generated'
            tag = f'{split}/{title}'
            if unique_id is not None:
                tag += f'_{unique_id}'

            self.summary_writer.add_image(tag,
                                          np.uint8(visuals_np),
                                          self.global_step,
                                          dataformats='HW')

        return

    def write(self, message, print_to_stdout=True):
        """Write a message to the log. If print_to_stdout is True, also print to stdout."""
        with open(self.log_filepath, 'a') as log_file:
            log_file.write(message + '\n')
        if print_to_stdout:
            print(message)

    def start_iter(self):
        """Log info for start of an iteration."""
        raise NotImplementedError

    def end_iter(self):
        """Log info for end of an iteration."""
        raise NotImplementedError

    def start_epoch(self):
        """Log info for start of an epoch."""
        raise NotImplementedError

    def end_epoch(self, metrics, curves):
        """Log info for end of an epoch. Save model parameters and update learning rate."""
        raise NotImplementedError
