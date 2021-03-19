"""
multi_task_classifier_3d_train_arg_parser.py
Arguments for training a multi-task 3d resnet classifier.
"""

from .train_arg_parser import TrainArgParser


class MTClassifier3DTrainArgParser(TrainArgParser):
    """Argument parser for args used only in train mode."""

    def __init__(self):
        super(MTClassifier3DTrainArgParser, self).__init__()

        self.parser.add_argument('--resnet3d_model', type=str, default='r3d_18', choices=(
            'r3d_18', 'mc3_18', 'r2plus1d_18'), help='resnet3d model to build for training.')
        self.parser.add_argument('--train_age_group', type=str, default='peds', choices=(
            'adult', 'peds'), help='Which dataset to train on.')
        self.parser.add_argument('--peds_csv_dir', type=str, default='/data2/SharonFolder/hikaru/peds_head_ct_numpy', help='Csv dir for peds data.')
        self.parser.add_argument('--adult_csv_dir', type=str, default='/data2/SharonFolder/hikaru/adult_head_ct_numpy', help='Csv dir for adult data.')
        self.parser.add_argument('--peds_features', nargs="+", type=str, default=['Bleed', 'Fracture', 'Tumor', 'Vent/EVD', 'Craniotomy', 'Normal'],
                                 help='Whitespace-separated list of features for peds.')
        self.parser.add_argument('--adult_features', nargs="+", type=str, default=['Intraparenchymal_hemorrhage', 'Fracture', 'Mass', 'Hydrocephalus_or_entrapment', 'Postoperative', 'NEGATIVE'],
                                 help='Whitespace-separated list of features for adults.')
        self.parser.add_argument('--num_slices', type=int, default=32, help='Number of slices from 3D data to sample from')
        self.parser.add_argument('--slice_size', type=int, default=512, help='Number of slices from 3D data to sample from')
        self.parser.add_argument('--focal_loss_alphas', nargs='+',
                                 default=None, type=float, help='Alpha hyperparameter of focal loss')
        # WarmupMultiStepLR args
        self.parser.add_argument('--lr-milestones', nargs='+',
                                 default=[20, 30, 40], type=int, help='decrease lr on milestones')
        self.parser.add_argument(
            '--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
        self.parser.add_argument(
            '--lr-warmup-epochs', default=10, type=int, help='number of warmup epochs')
