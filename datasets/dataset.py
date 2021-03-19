"""
dataset.py

Define simple dataset class for image data based on csv
"""

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


class Dataset(data.IterableDataset):
    """Dataset class for dicom images loaded on-demand"""

    def __init__(self, csv_path, split, toy, input_scan, output_scan, resample=(32, 400, 400), num_slices=32):

        """
        Args:
            csv_path (str or pathlib.Path): location of csv file
            split (str): train/val/test split
            toy (bool): Whether to use only 32 dicoms or all of them
            resample (int, int, int): Size of inputs after resampling transform.
            num_slices (int): number of (resampled) slices the model will look at
        """
        assert split in csv_path, (
            'Make separate file for each split to '
            f'ensure data is separated properly. Your {split} is using {csv_path}')
        print(f'Using dataset {csv_path} for {split}')

        self.csv_path = csv_path
        self.split = split
        self.toy = toy
        self.input_scan = input_scan
        self.goal_scan = output_scan

        df = pd.read_csv(csv_path)

        self.slices_per_example = num_slices
        self.resample = resample

        # self.filepaths = df['filepath'].tolist()
        example_groups = df.groupby(by='id').groups
        groups_as_df = [df.iloc[group] for group in example_groups.values()]
        grouped_as_numpy = [patient_df[['scan_type', 'filepath']].values for patient_df in groups_as_df]
        self.data_locations = [dict(numpy_group) for numpy_group in grouped_as_numpy]
        self.data_locations = [d for d in self.data_locations if input_scan in d.keys() and output_scan in d.keys()]
        if toy and split == 'train':
            self.data_locations = np.random.choice(self.data_locations, 32)

        self.transforms = self._set_transforms()

    def _set_transforms(self):
        """Decide transformations to data to be applied"""

        transforms_list_by_scan_type = {
            'con': lambda: ([
                custom_transforms.ToFloat(),
                custom_transforms.ResampleTo(self.resample),
                custom_transforms.Normalize(input_bounds=(-500, 500), pixel_mean=0.379),
                custom_transforms.EltListToBlockTensor(),
            ]),
            'noncon': lambda: ([
                custom_transforms.ToFloat(),
                custom_transforms.ResampleTo(self.resample),
                custom_transforms.Normalize(input_bounds=(-500, 500), pixel_mean=0.379),
                custom_transforms.EltListToBlockTensor(),
            ]),
            'cta': lambda: ([
                custom_transforms.ToFloat(),
                custom_transforms.ResampleTo(self.resample),
                custom_transforms.Normalize(input_bounds=(-500, 500), pixel_mean=0.271),
                custom_transforms.EltListToBlockTensor(),
            ]),
            'ctp': lambda: ([
                custom_transforms.ToFloat(),
                custom_transforms.ResampleTo((8, 256, 256)),
                # fixed resampling rate since CTPs are always 7-10, and don't need an extra argument
                custom_transforms.Normalize(input_bounds=(0, 60), pixel_mean=0.0158),
                custom_transforms.EltListToBlockTensor(),
            ])

        }

        transform_dict = dict((scan, transforms.Compose(transform_list_f())) for scan, transform_list_f in
                               transforms_list_by_scan_type.items() if scan in (self.input_scan, self.goal_scan))
        return transform_dict

    def __len__(self):
        """Required: specify dataset length for dataloader"""
        return len(self.data_locations)

    def _load_and_crop_scans(self, scan1_path, scan2_path):
        """
        Reads metadata for each scan about locations of the scan relative to some real-world scanner zero point,
        and then crops scans so they encompass the exact same volume of real-world space.

        Since these scans are done immediately after each other, this point can be considered fixed.
        This is usually done by a technician before the patient is scanned.
        
        Args:
            scan1_path: Path to scan 1
            scan2_path: Path to scan 2
        Returns:
            scan1, scan2: Cropped versions of each scan that have the same volume of space scanned.
        """
        np_dict_img1 = np.load(scan1_path)

        # If the following assert is hit, most likely pointing toward .npy files when should now be pointing to .npz
        assert np_dict_img1.__class__ != np.ndarray

        img1 = np_dict_img1['scan']
        img1_bounds = np_dict_img1['bounding_box_DHW']  # np array of shape 3x2 (D min-max, H min-max, W min-max)

        np_dict_img2 = np.load(scan2_path)
        img2 = np_dict_img2['scan']
        img2_bounds = np_dict_img2['bounding_box_DHW']

        final_bounds = get_final_bounding_box(img1_bounds, img2_bounds)
        img1 = crop_image_to_new_bounding_box(img1, img1_bounds, final_bounds)
        img2 = crop_image_to_new_bounding_box(img2, img2_bounds, final_bounds)

        return img1, img2

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

    def __getitem__(self, index):
        """Required: specify what each iteration in dataloader yields"""
        img_paths = self.data_locations[index]

        # TODO: do more than just extract con and noncon, maybe make a major adjustment
        input_path = img_paths[self.input_scan]
        goal_path = img_paths[self.goal_scan]

        inp, goal = self._load_and_crop_scans(input_path, goal_path)
        inp = self._preprocess_image_stack(inp, self.input_scan)
        goal = self._preprocess_image_stack(goal, self.goal_scan)

        return inp, goal

    def shuffle(self):
        random.shuffle(self.data_locations)


if __name__ == "__main__":
    print('Testing dummy debugging dataset')

    SPLIT = 'train'
    CSV_PATH = f'/data2/braingan/all/{SPLIT}.csv'
    TOY = True

    # These defaults will pull ~40 Gb of RAM, reduce one of them to get smaller impact.
    # Each example (of 128 x 512 x 512) will take up like 80Mb, take that into account.
    BS = 1
    NWORKERS = 4

    dataset = Dataset(csv_path=CSV_PATH,
                      split=SPLIT,
                      toy=TOY,
                      input_scan='cta',
                      output_scan='ctp',
                      resample=(128, 512, 512),
                      num_slices=128,
                      )

    loader = data.DataLoader(dataset,
                             batch_size=1,
                             num_workers=NWORKERS,
                             )


    def get_slice_ratios(loader):
        for inp, out in loader:
            inp_slices = inp.shape[2]
            out_slices = out.shape[2]
            print(inp_slices / out_slices, inp_slices, out_slices)


    def get_average_pixel_val(loader):
        # Calculate averages of dataset when only running Normalizer with no mean centering
        avg_inp = 0
        avg_out = 0
        n = len(loader)
        for inp, out in loader:
            avg_inp += np.average(inp) / n
            avg_out += np.average(out) / n
        print(avg_inp, avg_out)


    # get_slice_ratios(loader)
    get_average_pixel_val(loader)
