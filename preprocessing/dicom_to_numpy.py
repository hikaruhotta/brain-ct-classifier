"""
dicom_to_numpy.py
Preprocessing original dicom files into numpy.
Saving to separate clean folders for positives and controls.
"""

import os
from os.path import join, dirname

import numpy as np
from pydicom import dcmread, FileDataset
from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator
from pathlib import Path
from tqdm.contrib.concurrent import process_map
import pandas as pd


CT_SPACING = (5, .5, .5)
CTA_SPACING = (1.25, .5, .5)  # maybe use .488281 for both pixel spacings (instead of .5)
CTP_SPACING = (10, 1, 1)

# CT scan types
CON = 'con'
NONCON = 'noncon'
CTA = 'cta'
CTP = 'ctp'

# map from folder names (normalized) to CT types
FOLDER_TO_CT_TYPE = {
    "non enhanced brain": 'noncon',
    "enhanced brain": 'con',
    "head": "noncon",
    "head contrast": "con",
    "routine head standard": "noncon",
    "head w con": "con",
    "head w o": "noncon",
    "routine head": "noncon",
    "routine head w o": "noncon",
    "routine head std": "noncon",
    "de head post": "con",
    "post contrast head": "con",
    "post con head": "con",
    "non con head": "noncon",
    "axial head": "noncon",
    "emhanced head standard": "con",  # yes this is purposefully spelled incorrectly
    "post con": "con",
    "post con head routine": "con",
    "post contrast routine head": "con",
    "non con head std": "noncon",

    "head neck cta 1.25mm": "cta",
    "head neck cta": "cta",
    "h n angio 1.25mm": "cta",
    "h n angio": "cta",
    "1.25mm cta h n soft": "cta",
    "1.25mm h n angio": "cta",
    "1.25mm cta h n": "cta",
    "recon h n angio": "cta",
    "recon 2 h n angio": "cta",
    "cow": "cta",

    "rapid tmax colored": "tmax-color",
    "rapid tmax colored [s]": "tmax-color-s",
    "rapid color tmax": "tmax-color",
    "rapid color tmax [s]": "tmax-color-s",
    "rapid tmax": "tmax-gray",
    "rapid tmax [s]": "tmax-gray-s",
    "rapid perfusion parameter maps colored": "ctp-many"

}

KNOWN_OTHERS = {
    # CTA - We will use these names purely for logging purposes.
    # We will rely on on `ctaonly` directory for actually parsing cta files

    # Others in normal folders
    "ct head perfusion with contrast",

}

KNOWN_SUFFIXES = {
    "3.0 j30s 2",
    "1.0 i26f 2",
    "3.0 j30s 3 f 0.5",
    "5.0 j37s 2",
    "2.0 j37s 2",
    "4cc sec 70cc",
    "3.0 mpr ax",
    "1.0 mpr ax",
    "0.6 j37s 2",
    "3.0 i31f 2",
}

TMAX_Header_CTP = np.load(join('.', dirname(__file__), 'tmax_black_background.npy'))

POSITIVE = 'positive'
CONTROL = 'control'


def normalize_directory_name(directory: str):
    """Helper function for working with the many directory names used in our dataset.
    Uses lowercasing, removes dashes and underscores, and gets rid of a lot of suffixes that folders may
    contain. This ensures that the resulting folder will (hopefully) match with an entry in FOLDER_TO_CT_TYPE
    """
    normalized = directory.lower().replace("-", " ").replace("_", " ")
    for suffix in KNOWN_SUFFIXES:
        normalized = normalized.replace(suffix, "")
    return normalized.strip()


# converts a dicom scan to Hounsfield Units
def convert_to_hu_numpy(dcms, slope, intercept):
    np_volume = np.array([dcm.pixel_array for dcm in dcms])
    hu_scaled = np_volume.astype(np.float32) * slope + intercept
    hu_scaled = np.maximum(hu_scaled, -1000)
    return hu_scaled


def get_important_metadata_for_slices(dcm_slices):
    """
    Args:
        dcm_slices: python list of dcm slices returned by dcmread

    Returns:
        The slope, intercept, thicknesses, and spacings of slices.
        Will error if slope, intercept, or spacings change. Only thicknesses are allowed to change, and are returned as a list.
    """
    # TODO: Figure out whether to use slice thicknesses or differences in slice location as our numbers
    first = dcm_slices[0]
    slope = float(first.RescaleSlope)
    intercept = float(first.RescaleIntercept)
    locations = []
    spacings = []

    for dcm_slice in dcm_slices:
        pixel_spacing = [float(spacing) for spacing in dcm_slice.PixelSpacing]
        location = [float(dim) for dim in dcm_slice.ImagePositionPatient][
                   ::-1]  # was x,y,z. reversed is z,y,x like scan array.
        rescale_slope = float(dcm_slice.RescaleSlope)
        rescale_intercept = float(dcm_slice.RescaleIntercept)

        locations.append(location)
        spacings.append(pixel_spacing)
        assert rescale_slope == slope  # nor slope
        assert rescale_intercept == intercept  # nor intercept

    return slope, intercept, locations, spacings


def get_points_using_slice_location_and_pixel_spacing(locations: list, pixel_spacings: list, scan_shape: tuple):
    """ Returns a list of the coordinates of every pixel for a scan given the per-slice position of the scanner head
    and pixel spacings"""
    D, H, W = scan_shape
    num_points = D * H * W
    points = np.empty(shape=(3, num_points))
    assert D == len(locations) and D == len(pixel_spacings)
    for i, (location, pixel_spacing) in enumerate(zip(locations, pixel_spacings)):
        h, w = pixel_spacing
        grid = np.meshgrid([location[0]], [location[1] + h * i for i in range(H)],
                           [location[2] + w * i for i in range(W)], indexing='ij')
        for j in [0, 1, 2]:
            points[j, i * H * W:(i + 1) * H * W] = grid[j].flatten()
    return np.array(points).T


def get_axis_for_regular_grid_slice(locations, pixel_spacings, scan_shape):
    assert np.all([location[1] == locations[0][1] and location[2] == locations[0][2] for location in locations]) \
           and np.all([pixel_spacings[0] == pixel_spacing for pixel_spacing in pixel_spacings])
    D, H, W = scan_shape
    d0, h0, w0 = locations[0]
    h, w = pixel_spacings[0]
    depths = [location[0] for location in locations]
    heights = [h0 + i * h for i in range(H)]
    widths = [w0 + i * w for i in range(W)]

    return depths, heights, widths


def get_largest_internal_box(locations, pixel_spacings, scan_shape):
    D, H, W = scan_shape

    box_start_corner = (
        locations[0][0],
        max(l[1] for l in locations),
        max(l[2] for l in locations)
    )
    box_end_corner = (
        locations[-1][0],
        min(l[1] + ps[0] * (H - 1) for l, ps in zip(locations, pixel_spacings)),
        min(l[2] + ps[1] * (W - 1) for l, ps in zip(locations, pixel_spacings))
    )

    return box_start_corner, box_end_corner


def resample(np_volume: np.ndarray, locations: list, pixel_spacings, new_spacing):
    """

    Args:
        np_volume: (D x H x W) numpy.ndarray that represents all the dcm slices
        locations: location (z, y, x) for location of top left pixel in (D, H, W) dimensions for every slice
        pixel_spacing: difference in location of pixel centers (H, W)
        new_spacing: D, H, W output spacing. TODO: find good default for new spacing

    Returns:
        a resampled version of np_volume according to the new spacing.
    """

    box_start_corner, box_end_corner = get_largest_internal_box(locations, pixel_spacings, np_volume.shape)

    axis_points = [[start + step * i for i in range(int((end - start) // step))] for start, end, step in
                   zip(box_start_corner, box_end_corner, new_spacing)]

    assert np.all(np.array(box_end_corner) >= np.array(box_start_corner))

    # check if shifting locations or pixel-spacings causes a non-regular grid structure
    if np.all([location[1] == locations[0][1] and location[2] == locations[0][2] for location in locations]) \
            and np.all([pixel_spacings[0] == pixel_spacing for pixel_spacing in pixel_spacings]):
        points_as_axes = get_axis_for_regular_grid_slice(locations, pixel_spacings, np_volume.shape)
        interpolator = RegularGridInterpolator(points_as_axes, np_volume)
    else:
        # Cannot use regular grid interpolation since shifts happened.
        point_positions = get_points_using_slice_location_and_pixel_spacing(locations, pixel_spacings, np_volume.shape)
        interpolator = LinearNDInterpolator(point_positions, np_volume.flatten())

    ds, hs, ws = np.meshgrid(*axis_points, indexing='ij')
    output_points = np.array([ds.flatten(), hs.flatten(), ws.flatten()]).T

    new_volume_coords = [(min(axis), max(axis)) for axis in axis_points]
    new_volume = interpolator(output_points).reshape(ds.shape)

    return new_volume, new_volume_coords


def _read_dicoms_from_path(path):
    files = sorted([filename for filename in os.listdir(path) if '.dcm' in filename])
    assert len(files) > 0
    num = files[0].split('-')[1]
    if len(files[0].split('-')) > 3:
        # something was duplicated, no go
        return None
    paths = ['%s/%s' % (path, f) for f in files if num in f.split('-')[1]]  # only grab from first scan number
    return [dcmread(path) for path in paths]


def _sort_dicoms_by_location(dcms):
    sorted_by_location = sorted(dcms, key=lambda dcm: dcm.ImagePositionPatient[-1])
    return sorted_by_location


def _remove_dcms_with_duplicate_location(dcms):
    """
    At least one example of a scan with a few duplicate slices was found, so this function is built to remove slices
    that have an exact match to the instance number of an existing slice
    Args:
        dcms: loaded dicom files
    Returns:

    """
    slice_loc_set = set()
    filtered_dcms = []

    for dcm in dcms:
        if dcm.SliceLocation not in slice_loc_set:
            slice_loc_set.add(dcm.SliceLocation)
            filtered_dcms.append(dcm)
    return filtered_dcms


def read_sort_and_filter_dicoms(path):
    dcms = _read_dicoms_from_path(path)
    if dcms is None or np.any(['SliceLocation' not in dcm for dcm in dcms]):
        return None
    dcms = _remove_dcms_with_duplicate_location(dcms)
    dcms = _sort_dicoms_by_location(dcms)
    return dcms


def save_np_as_int16_with_metadata(save_loc, scan_as_3d_np: np.ndarray, metadata: dict):
    np.savez_compressed(save_loc, scan=np.round(scan_as_3d_np).astype(np.int16), **metadata)


def convert_dicom_stack_to_3d_numpy(path, output_spacing):
    dcms = read_sort_and_filter_dicoms(path)
    if dcms is None:
        return
    if len(dcms) < 20:
        print('patient at path %s has only %d dcms' % (path, len(dcms)))
        return

    slope, intercept, locations, spacings = get_important_metadata_for_slices(dcms)
    np_volume = convert_to_hu_numpy(dcms, slope, intercept)
    if not (np.all([location[1] == locations[0][1] and location[2] == locations[0][2] for location in locations])
            and np.all([spacings[0] == pixel_spacing for pixel_spacing in spacings])):
        print('problematic scan at %s ' % path)
        return

    np_volume, bounding_box = resample(np_volume, locations, spacings, output_spacing)
    return np_volume, bounding_box


# file name is like IM-0137-0001
def convert_dicom_stack_to_3d_numpy_and_save(path, save_loc, output_spacing):
    """
    Function called in parallel for loading dicoms, running preprocessing functions, and saving results as numpy array.
    THIS SHOULD ONLY BE CALLED ON CT con, CT noncon, and CTA. (CTP does not scale to HU, nor should it be resampled)
    Args:
        output_spacing: One of [CT_SPACING, CTA_SPACING], which is what the spacing between slices and pixels should be
        path: path to folder with dicom files
        save_loc: location of output numpy file, should be a string that does NOT include `.npz` extention
    Returns:

    """
    if os.path.exists(save_loc + '.npz'):
        return

    np_volume, bounding_box = convert_dicom_stack_to_3d_numpy(path, output_spacing)
    save_np_as_int16_with_metadata(save_loc, np_volume, {'bounding_box_DHW': bounding_box})


def convert_and_save_ctp_grayscale_dicom_stack(path, save_loc):
    dcms = read_sort_and_filter_dicoms(path)
    if dcms is None:
        return

    if os.path.exists(save_loc + '.npz'):
        return

    slope, intercept, locations, spacings = get_important_metadata_for_slices(dcms)
    np_volume = np.array([dcm.pixel_array for dcm in dcms])
    np_volume = np_volume * slope + intercept
    np_volume, bounding_box = resample(np_volume, locations, spacings, CTP_SPACING)
    save_np_as_int16_with_metadata(save_loc, np_volume, {'bounding_box_DHW': bounding_box})


def run_tmax_color_to_grayscale_model(color_dicoms, s=False):
    """

    Args:
        color_folder_path: path to folder holding Tmax colored dicom files
        s: whether the folder is using [s] units. This is denoted by folder path containing _[s] at the end.

    """

    # TODO: Remove scale on left side
    # TODO: Add color-->grayscale model here

    return None


def convert_ct(current_path: Path, ct_type: str, output_directory):
    """
    Converts a single CT scan (which is a set of multiple .dcm slices within 1 folder) to a numpy object.
    This method works for 'con', 'noncon', 'cta', and 'ctp'
    Args:
        current_path: path to folder with CT scan
            for con, noncon, cta this directly points at folder with dcm files
            for ctp, this points toward the patient scan directory, within which there could be multiple valid
            ctp outputs (for example: Tmax grayscale, Tmax color, or a set of many CTP outputs which Tmax should be parsed from)
        ct_type: one of (con, noncon, cta, ctp)
        output_directory: directory where to put generated numpy results
    """
    if ct_type == CON or ct_type == NONCON or ct_type == CTA:
        id_of_folder = int(current_path.parent.name)
        patient_name = current_path.parent.parent.name
        save_loc = '%s/%s--%d--%s' % (output_directory, patient_name, id_of_folder, ct_type)
        output_spacing = CT_SPACING if ct_type != CTA else CTA_SPACING
        convert_dicom_stack_to_3d_numpy_and_save(current_path, save_loc, output_spacing)

    if ct_type == CTP:
        id_of_folder = int(current_path.name[:current_path.name.index('_')])  # CTP have date after id in folder name
        patient_name = current_path.parent.name
        save_loc = '%s/%s--%d--%s' % (output_directory, patient_name, id_of_folder, ct_type)
        ctp_dirs = {}
        for d in os.listdir(current_path):
            normalized = normalize_directory_name(d)
            if normalized in FOLDER_TO_CT_TYPE.keys():
                if FOLDER_TO_CT_TYPE[normalized] in ctp_dirs:
                    print("WARN: Multiple directories of same CTP scan type!!")
                ctp_dirs[FOLDER_TO_CT_TYPE[normalized]] = d

        if 'tmax-gray' in ctp_dirs.keys():
            tmax_path = current_path.joinpath(ctp_dirs['tmax-gray'])
            convert_and_save_ctp_grayscale_dicom_stack(tmax_path, save_loc)
        if 'tmax-gray-s' in ctp_dirs.keys():
            tmax_path = current_path.joinpath(ctp_dirs['tmax-gray-s'])
            convert_and_save_ctp_grayscale_dicom_stack(tmax_path, save_loc + '-s')
        else:
            print('Folder %s did not have any gray Tmax data' % current_path)


# top directory should be /data2/SharonFolder/lvo/raw/######
def convert_and_save_noncon_con_cta(top_directory, output_directory, print_abnormalities=True):
    name_of_top_dir = Path(top_directory).name
    paths = []
    ct_types = []

    for current_path, _, current_file_names in os.walk(top_directory, topdown=True):

        current_path = Path(current_path)

        # Want to be in {{type of scan}} folder, so ignore any files/folders that are not 3 layers in
        # i.e. top directory = /data2/SharonFolder/lvo/raw/control/
        # and path to pass to convert_ct is /data2/SharonFolder/lvo/raw/control/{{patient}}/{{id}}/{{type of scan}}
        parents = [parent.name for parent in current_path.parents]
        if parents[2] != name_of_top_dir:
            continue

        folder_name = normalize_directory_name(current_path.name)
        if folder_name in FOLDER_TO_CT_TYPE.keys():
            paths.append(current_path)
            ct_types.append(FOLDER_TO_CT_TYPE[folder_name])
        elif folder_name not in KNOWN_OTHERS and print_abnormalities:
            print('Path %s has no matches, folder %s' % (current_path, folder_name))

    process_map(convert_ct, paths, ct_types, [output_directory for _ in paths], max_workers=6, chunksize=1)


def convert_and_save_ctp(top_directory, output_directory, print_abnormalities=True):
    name_of_top_dir = Path(top_directory).name

    paths = []

    for current_path, _, current_file_names in os.walk(top_directory, topdown=True):

        current_path = Path(current_path)

        # Want to be in {{medical id}} folder, so ignore any files/folders that are not 3 layers in
        # i.e. top directory = /data2/SharonFolder/CTP/LVO_CTP/Control/
        # convert_ct called on /data2/SharonFolder/CTP/LVO_CTP/Control/Control_CTP1/{{patient}}/{{id}}_{{date}}/
        parents = [parent.name for parent in current_path.parents]
        if parents[2] != name_of_top_dir:
            continue
        paths.append(current_path)
        # convert_ct(current_path, 'ctp', output_directory)

    process_map(convert_ct, paths, [CTP for _ in paths], [output_directory for _ in paths], max_workers=4,
                chunksize=1)


def convert_and_save(remove_and_replace=False, debug=True):
    """
    Args:
        remove_and_replace (bool): Whether all files within output folder are deleted before running conversion
        debug (bool): Whether to use "temp_testdir" as output directory or "lvo-numpy"
    """
    sources = {
        'noncon-con-cta': {
            POSITIVE: '/data2/SharonFolder/lvo/raw/positive/',
            CONTROL: '/data2/SharonFolder/lvo/raw/control/'
        },
        'ctp': {
            POSITIVE: '/data2/SharonFolder/CTP/LVO_CTP/LVO/',
            CONTROL: '/data2/SharonFolder/CTP/LVO_CTP/Control/'
        }
    }

    if debug:
        destination_root = '/data2/SharonFolder/temp_testdir/'
    else:
        destination_root = '/data2/SharonFolder/lvo-numpy/'

    patient_groups = [POSITIVE, CONTROL]  # If we want more patient groups, they will get their own folders
    output_folders = [destination_root + patient_group for patient_group in patient_groups]

    if remove_and_replace:
        # Delete all existing files in output folders
        for output_folder in output_folders:
            for f in os.listdir(output_folder):
                try:
                    os.remove(output_folder + '/' + f)
                except:
                    continue

    for patient_group, output_folder in zip(patient_groups, output_folders):
        print('patient_group: %s, noncon-con-cta' % patient_group)
        convert_and_save_noncon_con_cta(sources['noncon-con-cta'][patient_group], output_folder,
                                        print_abnormalities=True)
        print('patient_group: %s, ctp' % patient_group)
        convert_and_save_ctp(sources['ctp'][patient_group], output_folder, print_abnormalities=True)


if __name__ == "__main__":
    convert_and_save(remove_and_replace=False, debug=True)
