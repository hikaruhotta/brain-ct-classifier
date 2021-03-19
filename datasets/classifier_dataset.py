"""
dataset.py

Define simple dataset class for image data based on csv
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import math
import random
from matplotlib import pyplot as plt

import datasets.transforms as custom_transforms
from datasets.dataset_utils import get_final_bounding_box, crop_image_to_new_bounding_box


class ClassifierDataset(data.Dataset):
    """Dataset class for dicom images loaded on-demand"""

    def __init__(self, csv_dir, split, label_cols, toy=False, input_scan='noncon', resample=(32, 400, 400)):

        """
        Args:
            csv_dir (str or pathlib.Path): location of csv file
            split (str): train/val/test split
            toy (bool): Whether to use only 32 dicoms or all of them
            resample (int, int, int): Size of inputs after resampling transform.
            num_slices (int): number of (resampled) slices the model will look at
        """
        # assert split in csv_path, (
        #     'Make separate file for each split to '
        #     f'ensure data is separated properly. Your {split} is using {csv_path}')
        print(f'Using dataset {csv_dir} for {split}')

        self.csv_path = os.path.join(csv_dir, f"{split}.csv")
        self.split = split
        self.toy = toy
        self.input_scan = input_scan
        self.label_cols = label_cols

        df = pd.read_csv(self.csv_path)

        self.resample = resample

        filepaths = df['filepath'].values
        scan_types = df['scan_type'].values

        filepaths = filepaths[np.where(scan_types == input_scan)]
        self.data_locations = filepaths
        if toy and split == 'train':
            self.data_locations = np.random.choice(self.data_locations, 32)

        self.transforms = self._set_transforms()

    def _set_transforms(self):
        """Decide transformations to data to be applied"""

        transforms_list_by_scan_type = {
            'noncon': lambda: ([
                custom_transforms.ToFloat(),
                custom_transforms.ResampleTo(self.resample),
                custom_transforms.Normalize(input_bounds=(-500, 500), pixel_mean=0.379),
                custom_transforms.EltListToBlockTensor(),
                transforms.RandomAffine(degrees=15, translate=(0.05, 0.05)),
                transforms.RandomHorizontalFlip(p=0.5),
            ])

        }

        transform_dict = dict((scan, transforms.Compose(transform_list_f())) for scan, transform_list_f in
                              transforms_list_by_scan_type.items())
        return transform_dict

    def __len__(self):
        """Required: specify dataset length for dataloader"""
        return len(self.data_locations)

    def _preprocess_image_stack(self, img, scan_type):
        img = self.transforms[scan_type](img)
        return img

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        total_len = len(self.data_locations)
        if worker_info is None:  # single-process data loading, return the full iterator
            iter_start = 0
            iter_end = total_len
        else:  # in a worker process
            # split workload
            per_worker = int(math.ceil(total_len / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, total_len)
        return (self[i] for i in range(iter_start, iter_end))
    
    # TODO: Revisit
    def normalize_label(self, x):
        if type(x) == str:
            if x.isnumeric():
                x = float(x)
            else:
                return 0
        if math.isnan(x) or x < 0:
            return 0.
        elif x > 1:
            return 1.
        else:
            return x

    def __getitem__(self, index):
        """Required: specify what each iteration in dataloader yields"""
        npz_path = self.data_locations[index]
        npz_dict = np.load(npz_path, allow_pickle=True)
        inp = self._preprocess_image_stack(npz_dict['scan'], self.input_scan)
        record = npz_dict['record'].item()
        label_list = [record[col] for col in self.label_cols]
        label_list = [self.normalize_label(label) for label in label_list]
        label = torch.tensor(label_list, dtype=torch.float32)
        return inp, label

    def shuffle(self):
        random.shuffle(self.data_locations)


if __name__ == "__main__":
    print('Testing dummy debugging dataset')

    SPLIT = 'train'
    CSV_PATH = f'/data2/SharonFolder/hikaru/peds_head_ct_numpy/{SPLIT}.csv'
    TOY = False

    # These defaults will pull ~40 Gb of RAM, reduce one of them to get smaller impact.
    # Each example (of 128 x 512 x 512) will take up like 80Mb, take that into account.
    BS = 1
    NWORKERS = 4

    dataset = ClassifierDataset(csv_path=CSV_PATH,
                                split=SPLIT,
                                toy=TOY,
                                label_cols=['Bleed', 'Fracture', 'Tumor']
                                )

    loader = data.DataLoader(dataset,
                             batch_size=1,
                             num_workers=NWORKERS,
                             )

    # for scan, label in loader:
    #     print(scan)
    #     print(label)
    #     break

    for scan, label in loader:
        continue
