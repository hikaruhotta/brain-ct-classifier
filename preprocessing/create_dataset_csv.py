"""
create_dataset_csv.py
Organizing of numpy files into a csv file with locations, some metadata, etc.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import time


CLASS_FOLDERS = ['control', 'positive']
CSV_COLS = ["patient", "patient_group", "scan_type", "id", "filepath"]

OUT_FOLDERS = {
    'noncon-con': Path('/data2/braingan/debug'),
    'cta-ctp': Path('/data2/braingan/cta-ctp'),
    'all': Path('/data2/braingan/all'),
    'debug-all': Path('/data2/braingan/debug-all')
}


def get_data_for_file(numpy_file_path: Path):
    """
    Obtains every piece of data to be placed into train/val/test csvs for a specific numpy file.
    Args:
        numpy_file_path: path to numpy file on a patient
    """
    file_class = numpy_file_path.parent.name
    # check that the file is in our known classes
    if file_class not in CLASS_FOLDERS:
        print('No class found for %s' % (str(numpy_file_path)))
        return None

    # name of current file, looks like First_Middle_Last--{type}
    file_name = numpy_file_path.name.replace('.npy', '').replace('.npz', '') # npz is the new path type, npy old.
    person, id, scan_type = file_name.split('--')  # use 2 dashes since some people have a dash in their name

    return {
        "patient": person,
        "patient_group": file_class,
        "scan_type": scan_type,
        "id": id,
        "filepath": str(numpy_file_path),
    }


def create_dataframe_from_numpy(root_numpy_folder):
    """
    @type root_numpy_folder: str
    Folder location that has "control" and "positive" folders containing numpy examples
    """

    root = Path(root_numpy_folder)
    numpy_paths = root.glob("**/*.np[yz]")
    data = (get_data_for_file(numpy_path) for numpy_path in numpy_paths)
    df = pd.DataFrame(data=data, columns=CSV_COLS)
    return df


def remove_patients_without_enough_scans(df: pd.DataFrame, num_scans, verbose=False):
    """
    All unpaired data is removed from dataframe. Assumption is that input data should have at most pairs of scans.
    TODO: This may be modified later to have an argument passed in for the number of scans a patient should have.
    @param verbose:
    @param df:
    @return:
    """
    grouped_by_patient = df[['patient', "patient_group", "id"]].groupby(by='id').count()
    unpaired = grouped_by_patient.loc[grouped_by_patient['patient_group'] < num_scans]
    print('Removing %d patients due to lack of pair' % unpaired.size)
    if verbose:
        print(unpaired)
    unpaired_mask = [patient not in unpaired.index.values for patient in df['id'].values]
    return df.loc[unpaired_mask]


def remove_patients_with_large_slice_number_mismatch(df: pd.DataFrame, max_difference=1, verbose=False):
    """
    If any examples have slice numbers that are more than max_difference apart, remove them.
    Dicom slices have already been normalized and reshaped into a volume by the time this method is called,
     which should be roughly the same for the same patient.
    @param df:
    @return:
    """

    def slice_mismatch(series):
        filepaths = series.values
        slices = [np.load(filepath)['scan'].shape[0] for filepath in filepaths]
        return np.max(slices) - np.min(slices) > max_difference

    grouped_by_patient = df[["id", "filepath"]].groupby(by='id').agg(func=slice_mismatch)
    slices_dont_match = grouped_by_patient.loc[grouped_by_patient['filepath']]
    print('Removing %d patients due to unmatched slice numbers' % slices_dont_match.size)
    if verbose:
        print(slices_dont_match)
    slices_dont_match_mask = [patient not in slices_dont_match.index.values for patient in df['patient'].values]
    return df.loc[slices_dont_match_mask]


def create_sliced_csvs(df: pd.DataFrame, val_ratio=.1, test_ratio=.1):
    """
    Splits df into train, val, test based on randomly selected patients.
    TODO: Instead of random, a stratified split on positive/control would be good. Current sets are pretty close.
    @param df:
    @param val_ratio:
    @param test_ratio:
    @return:
    """
    ids = set(df['id'].values)
    num_val = int(len(ids) * val_ratio)
    num_test = int(len(ids) * test_ratio)
    non_train_ids = np.random.choice(list(ids), num_val + num_test, replace=False)
    val_ids = set(non_train_ids[:num_val])
    test_ids = set(non_train_ids[num_val:])
    train_ids = ids - val_ids - test_ids

    train_df = df.loc[[patient in train_ids for patient in df['id'].values]]
    val_df = df.loc[[patient in val_ids for patient in df['id'].values]]
    test_df = df.loc[[patient in test_ids for patient in df['id'].values]]

    if 'patient_group' in df.columns and 'patient' in df.columns:
        train_df = train_df.set_index(['patient', 'patient_group', 'id']).sort_index()
        val_df = val_df.set_index(['patient', 'patient_group', 'id']).sort_index()
        test_df = test_df.set_index(['patient', 'patient_group', 'id']).sort_index()
    else:
        train_df = train_df.set_index('id').sort_index()
        val_df = val_df.set_index('id').sort_index()
        test_df = test_df.set_index('id').sort_index()

    return train_df, val_df, test_df


def create_cta_to_ctp_df(original_df: pd.DataFrame, out_folder):
    cta = original_df[original_df.scan_type.values == 'cta']
    ctp = original_df[original_df.scan_type.values == 'ctp']
    df = cta.append(ctp)
    df = remove_patients_without_enough_scans(df, 2)

    train, val, test = create_sliced_csvs(df)
    train.to_csv(out_folder.joinpath('train.csv'))
    val.to_csv(out_folder.joinpath('val.csv'))
    test.to_csv(out_folder.joinpath('test.csv'))

    print('Successfully created cta-ctp dataset pairs')
    # TODO: When we have almost every patient with CTP data (right now only 100 patients total have grayscale) we
    #  should move both this and noncon_to_con back into 1 folder and remove examples that are missing any 1 of the 4
    #  scans.


def create_noncon_to_con_df(original_df: pd.DataFrame, out_folder: Path):
    noncon = original_df[original_df.scan_type.values == 'noncon']
    con = original_df[original_df.scan_type.values == 'con']
    df = noncon.append(con)
    df = remove_patients_without_enough_scans(df, 2)
    df = remove_patients_with_large_slice_number_mismatch(df, verbose=True)

    train, val, test = create_sliced_csvs(df)
    train.to_csv(out_folder.joinpath('train.csv'))
    val.to_csv(out_folder.joinpath('val.csv'))
    test.to_csv(out_folder.joinpath('test.csv'))

    print('Successfully created noncon-con dataset pairs')


def create_df_with_all_scans(original_df: pd.DataFrame, out_folder, remove_examples_missing_scans=False):
    df = original_df
    if remove_examples_missing_scans:
        num_scan_types = len(set(original_df.scan_type.values))
        df = remove_patients_without_enough_scans(df, num_scan_types)
    train, val, test = create_sliced_csvs(df)
    train.to_csv(out_folder.joinpath('train.csv'))
    val.to_csv(out_folder.joinpath('val.csv'))
    test.to_csv(out_folder.joinpath('test.csv'))

    print('Successfully created noncon-con dataset pairs')


if __name__ == "__main__":
    debug = True

    if debug:
        df: pd.DataFrame = create_dataframe_from_numpy('/data2/SharonFolder/temp_testdir/')
        print('debug-all')
        start = time.time()
        create_df_with_all_scans(df, OUT_FOLDERS['debug-all'], remove_examples_missing_scans=False)
        print('finished in %d sec' % int(time.time() - start))
    else:
        df: pd.DataFrame = create_dataframe_from_numpy('/data2/SharonFolder/lvo-numpy/')
        print('noncon --> con')
        start = time.time()
        create_noncon_to_con_df(df, OUT_FOLDERS['noncon-con'])
        print('finished in %d sec' % int(time.time() - start))

        print('cta --> ctp')
        start = time.time()
        create_cta_to_ctp_df(df, OUT_FOLDERS['cta-ctp'])
        print('finished in %d sec' % int(time.time() - start))

        print('all')
        start = time.time()
        create_df_with_all_scans(df, OUT_FOLDERS['all'], remove_examples_missing_scans=False)
        print('finished in %d sec' % int(time.time() - start))
