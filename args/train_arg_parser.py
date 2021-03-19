"""
train_arg_parser.py
Arguments for training
"""

from .base_arg_parser import BaseArgParser


class TrainArgParser(BaseArgParser):
    """Argument parser for args used only in train mode."""
    def __init__(self):
        super(TrainArgParser, self).__init__()
        
        self.parser.add_argument('--split', type=str, default='train', help='Split/phase: training if using this arg parser')
        self.parser.add_argument('--start_epoch', type=int, default=1, help='Epoch to start training from.')
        self.parser.add_argument('--num_epochs', type=int, default=12, help='Number of epochs to train.')
        self.parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate.')
        self.parser.add_argument('--reconstruction_loss_weight', type=float, default=0, help='Weight of the reconstruction loss (original Pix2Pix paper used default 100)')
        self.parser.add_argument('--toy', default=False, action='store_true', help='Use toy (subset of) dataset to train on')
        self.parser.add_argument('--step_train_discriminator', type=float, default=1, help='Train discriminator every x steps. Set x here.')
        
        self.parser.add_argument('--beta1', type=float, default=0.5, help='Beta1 value for Adam optimizer.')

        self.parser.add_argument('--max_ckpts', type=int, default=15, help='Max ckpts to save.')
        self.parser.add_argument('--continue_train', default=False, action='store_true', help='Continue training from last epoch.')
        self.parser.add_argument('--load_path', type=str, default=None, help='Load from a previous checkpoint.')
        self.parser.add_argument('--best_ckpt_metric', type=str, default='validation_f1', help='Load from a previous checkpoint.')
        self.parser.add_argument('--maximize_metric', default=False, action='store_true', help='Maximize to best ckpt metric.')
        
        self.parser.add_argument('--steps_per_print', type=int, default=10, help='Steps taken for each print of logger')
        self.parser.add_argument('--steps_per_visual', type=int, default=100, help='Steps for  each visual to be printed by logger in tb')
        self.parser.add_argument('--epochs_per_save', type=int, default=10, help='Save model every n epochs.')
        self.parser.add_argument('--epochs_per_eval', type=int, default=5, help='Evaluate model every n epochs.')
