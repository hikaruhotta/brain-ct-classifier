import torch

from time import time
from torchnet.meter import AverageValueMeter

from .base_logger import BaseLogger


class TrainLogger(BaseLogger):
    """Class for logging training info to the console and saving model parameters to disk."""

    def __init__(self, args, dataset_len, phase=None):
        super(TrainLogger, self).__init__(args, dataset_len)
       
        # Tag suffix used for indicating training phase in loss + viz
        self.tag_suffix = phase
        
        self.num_epochs = args.num_epochs
        self.split = args.split

        self.gen_loss_meter = AverageValueMeter()
        self.disc_loss_meter = AverageValueMeter()
        
        self.device = args.device
        self.iter = 0
        self.steps_per_print = args.steps_per_print
        self.steps_per_visual = args.steps_per_visual

    def log_hparams(self, args):
        """Log all the hyper parameters in tensorboard"""

        hparams = {}
        args_dict = vars(args)
        for key in args_dict:
            hparams.update({'hparams/' + key: args_dict[key]})

        self._log_text(hparams)

    def start_iter(self):
        """Log info for start of an iteration."""
        self.iter_start_time = time()

    def log_iter(self, inputs, targets, generated, gen_loss, disc_loss):
        """Log results from a training iteration."""
        batch_size = inputs.size(0)

        gen_loss = gen_loss.item()
        self.gen_loss_meter.add(gen_loss, batch_size)

        disc_loss = disc_loss.item()
        self.disc_loss_meter.add(disc_loss, batch_size)

        # Periodically write to the log and TensorBoard
        if self.iter % self.steps_per_print == 0:

            # Write a header for the log entry
            avg_time = (time() - self.iter_start_time) / batch_size
            message = f"[epoch: {self.epoch}, iter: {self.iter} / {self.dataset_len}, time: {avg_time:.2f}, gen loss: {self.gen_loss_meter.mean:.3g}, disc loss: {self.disc_loss_meter.mean:.3g}]"
            self.write(message)

            # Write all errors as scalars to the graph
            self._log_scalars({'Loss_Gen': self.gen_loss_meter.mean},
                              print_to_stdout=False,
                              unique_id=self.tag_suffix)
            self.gen_loss_meter.reset()

            self._log_scalars({'Loss_Disc': self.disc_loss_meter.mean},
                              print_to_stdout=False,
                              unique_id=self.tag_suffix)
            self.disc_loss_meter.reset()

        # Periodically visualize up to num_visuals training examples from the batch
        if self.iter % self.steps_per_visual == 0:
            self.visualize(inputs, targets, generated, self.split, unique_id=self.tag_suffix)

    def log_metrics(self, metrics):
        self._log_scalars(metrics)

    def end_iter(self):
        """Log info for end of an iteration."""
        self.iter += self.batch_size
        self.global_step += self.batch_size

    def start_epoch(self):
        """Log info for start of an epoch."""
        self.epoch_start_time = time()
        self.iter = 0
        self.write(f'[start of epoch {self.epoch}]')

    def end_epoch(self):
        """Log info for end of an epoch."""
        epoch_time = time() - self.epoch_start_time
        self.write(f'[end of epoch {self.epoch}, epoch time: {epoch_time:.2g}]')
        self.epoch += 1
        
    def is_finished_training(self):
        """Return True if finished training, otherwise return False."""
        return 0 < self.num_epochs < self.epoch

