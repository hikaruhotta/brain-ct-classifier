"""
loader.py

Initialize torch dataloader based on specified args for the appropriate dataset class
"""

from pathlib import Path
import torch.utils.data as data

from datasets.dataset import Dataset


def get_loader_from_args(args):
    return get_loader(args.csv_dir, args.split, args.resample, args.slices, args.batch_size, args.num_workers,
                      args.toy, args.input_scan, args.goal_scan)


def get_loader(csv_dir, split, resample, slices_per_example, batch_size, num_workers, toy, input_scan, output_scan):
    """Initialize the data loader"""
    csv_dir = Path(csv_dir)

    # Default csv path is csv_dir / train.csv or whatever split is
    csv_path = str(csv_dir / f'{split}.csv')

    dataset = Dataset(csv_path=csv_path,
                      split=split,
                      toy=toy,
                      input_scan=input_scan,
                      output_scan=output_scan,
                      resample=resample,
                      num_slices=slices_per_example)
    loader = data.DataLoader(dataset, batch_size=batch_size, drop_last=False, pin_memory=True, num_workers=num_workers)

    return loader


if __name__ == "__main__":
    print('Testing dummy debugging loader')

    from args import TrainArgParser
    parser = TrainArgParser()
    args = parser.parse_args()

    loader = get_loader_from_args(args)

    item = next(iter(loader))
    print('First item', item.shape, item)
