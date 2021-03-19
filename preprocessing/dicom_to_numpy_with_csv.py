"""
dicom_to_numpy_with_csv.py
Preprocessing original dicom files into numpy.
Saves each entry with associated record from csv file in .npz directory
"""
import os
from pathlib import Path
from tqdm.contrib.concurrent import process_map
import pandas as pd

from preprocessing.create_dataset_csv import create_sliced_csvs
from preprocessing.dicom_to_numpy import convert_dicom_stack_to_3d_numpy, CT_SPACING, save_np_as_int16_with_metadata
import numpy as np


CT_TYPE_MAP = {
    "CT HEAD": "noncon",
    "LPCH CT HEAD WO CONTRAST 70450": "noncon",
    # TODO: Check on whether we can use (and what type of scan is) "CT HEAD CSPINE", "CT HEAD FACIAL BONES CSPINE", "CT HEAD REFERENCE ONLY"
}

CT_TYPE_COL = 'Modality'


def convert_one_folder(path: Path, id, record, output_dir):
    if output_dir.joinpath(str(id) + '.npz').exists():
        return True

    save_loc = output_dir.joinpath(str(id))
    output_spacing = CT_SPACING  # TODO: If any CTAs in these libraries, dynamically choose spacing? Or maybe set all to CT spacing for reduce volume size.

    output = convert_dicom_stack_to_3d_numpy(path, output_spacing)
    if output is not None:
        np_volume, _ = output
        save_np_as_int16_with_metadata(save_loc, np_volume, {'record': record})
        return True
    return False


def convert_and_save_folder_with_csv(root_dir, csv_path, id_column, output_dir, id_loc_in_path_reversed=1, depth_of_series_folder=2, encoding='utf-8'):
    # id_column is "Accession Number" for Peds, "Acc" for Adults iirc

    name_of_top_dir = Path(root_dir).name
    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir()

    def clean_column_name(name: str):
        name = name.replace(',', ' ')
        name = name.replace('  ', ' ')
        name = name.strip()
        name = name.replace(' ', '_')
        return name

    paths = []
    ids = []
    records_list = []
    df = pd.read_csv(csv_path, encoding=encoding)
    df = df.set_index(id_column)
    df.columns = [clean_column_name(col_name) for col_name in df.columns]
    records = df.to_dict('index')

    for current_path, _, current_file_names in os.walk(root_dir, topdown=True):

        current_path = Path(current_path)

        # Want to be in {{medical id}} folder, so ignore any files/folders that are not 3 layers in
        # i.e. top directory = /data2/SharonFolder/CTP/LVO_CTP/Control/
        # convert_ct called on /data2/SharonFolder/CTP/LVO_CTP/Control/Control_CTP1/{{patient}}/{{id}}_{{date}}/
        parents = [parent.name for parent in current_path.parents]
        if parents[depth_of_series_folder - 1] != name_of_top_dir:
            continue
        if id_loc_in_path_reversed > 0:
            id = int(current_path.parents[id_loc_in_path_reversed - 1].name)
        else:
            id = int(current_path.name)

        if id in records:
            paths.append(current_path)
            ids.append(id)
            records_list.append(records[id])
        # convert_one_folder(current_path, id, records[id], output_dir)

    saved_bools = process_map(convert_one_folder, paths, ids, records_list, [output_dir for _ in paths], max_workers=4, chunksize=4)
    unsaved_ids = np.array(ids)[(1 - np.array(saved_bools)).astype(np.bool)]
    for id in unsaved_ids:
        del records[id]
    create_csvs(records, output_dir, output_dir)


def create_csvs(records, output_dir, csv_output_loc):
    output_dir = Path(output_dir)
    csv_output_loc = Path(csv_output_loc)
    df = pd.DataFrame(columns=['id', 'scan_type', 'filepath'])
    df_list = []
    for id, record in records.items():
        npz_loc = output_dir.joinpath(str(id) + '.npz')
        if record[CT_TYPE_COL] in CT_TYPE_MAP and npz_loc.exists():
            ct_type = CT_TYPE_MAP[record[CT_TYPE_COL]]
            df_list.append({
                'id': id,
                'scan_type': ct_type,
                'filepath': npz_loc
            })
    df = df.append(df_list)
    train, val, test = create_sliced_csvs(df)
    train.to_csv(csv_output_loc.joinpath('train.csv'))
    val.to_csv(csv_output_loc.joinpath('val.csv'))
    test.to_csv(csv_output_loc.joinpath('test.csv'))
    df = df.set_index('id').sort_index()
    df.to_csv(csv_output_loc.joinpath('all.csv'))


if __name__ == "__main__":
    # convert_and_save_folder_with_csv(
    #     root_dir='/data2/SharonFolder/PedsHeadCT/patient',
    #     csv_path='/data2/SharonFolder/PedsHeadCT/PedsHeadCT_summary_fixed.csv',
    #     id_column='Accession Number',
    #     output_dir='/data2/SharonFolder/hikaru/peds_head_ct_numpy',
    # )
    convert_and_save_folder_with_csv(
        root_dir='/data2/SharonFolder/AdultHeadCT',
        csv_path='/data2/SharonFolder/AdultHeadCT/AdultHeadCT_labels.csv',
        id_column='Acc',
        output_dir='/data2/SharonFolder/hikaru/adult_head_ct_numpy',
        encoding='ISO-8859-1'
    )
