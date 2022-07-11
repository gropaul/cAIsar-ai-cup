

from data_processing.data_processing_functions import annotate_all_files
from utils.SYSCONFIG import DATA_PATH
from typing import List, Tuple
import pandas as pd
import os
import json

from utils.SYSCONFIG import DATA_PATH, PLATFORM, DATA_GAIT,DATA_SUBMISSION
from utils.util_functions import printc, CustomBar, get_folder_size
from utils.path_utils import create_index_for_dir

import data_generator.batch_data_path_generator as pg
from cup_scripts.utils import load_test

def create_submission_dataset():
    
    def send_message(message: str, **kwargs) -> None:
        printc(source='[SUBMISSION DATASET CREATOR]', message=message, **kwargs)


    send_message(f"Starting to download evaluation data ...")

    # 3 Submission Data

    send_message('Loading evaluation data ...')
    X_test, metadata_test = load_test()
    send_message('Loading data completed, checking data.')

    # check test data validity using meta data
    for i in range(0, len(metadata_test), 2):

        x0, x1 = metadata_test[i].copy(), metadata_test[i + 1].copy()

        del x0['SensorLocation']
        del x1['SensorLocation']

        if x0 != x1:
            send_message(f"Mismatch for {i} and {i+1}")

    send_message('All checks completed.')

    submission_data_folder = pg.get_original_folder_path(
        data_subset_name=DATA_SUBMISSION)

    names = X_test[0].columns.values
    left_name_mapper = {name: 'L'+name for name in names}
    right_name_mapper = {name: 'R'+name for name in names}

    send_message('formatting data ...')

    # merging two tables together and saving csv files
    for i in range(0, len(X_test), 2):
        x0 = X_test[i]
        x1 = X_test[i+1]

        lx0 = x0.rename(columns=left_name_mapper)
        rx1 = x1.rename(columns=right_name_mapper)

        n_rows = len(x0)
        zero_labels = pd.DataFrame()
        zero_labels['LFA'] = [0] * n_rows
        zero_labels['RFA'] = [0] * n_rows

        merged = pd.concat([lx0, rx1, zero_labels], axis=1)

        # file format needed for sorting
        file_path_csv = os.path.join(submission_data_folder, f'{i}-{i}.csv')
        merged.to_csv(file_path_csv,index=False)


        meta0 = metadata_test[i]
        file_path_json = os.path.join(submission_data_folder, f'{i}-{i}.json')
        with open(file_path_json, 'w', encoding='utf8') as json_file:
            json.dump(meta0, json_file, ensure_ascii=False)
        

    send_message("done creating & formatting data. Creating data index.")

    submission_csv_index_path = pg.get_original_csv_index_path(
        data_subset_name=DATA_SUBMISSION)

    submission_data_folder = pg.get_original_folder_path(
        data_subset_name=DATA_SUBMISSION)

    submission_data_index = create_index_for_dir(
        dir_path=submission_data_folder,
        index_path=submission_csv_index_path,
        file_ending='.csv',
        save=True,
        sort=True
    )

    submission_json_index_path = pg.get_original_json_index_path(data_subset_name=DATA_SUBMISSION)

    submission_data_index = create_index_for_dir(
        dir_path=submission_data_folder,
        index_path=submission_json_index_path,
        file_ending='.json',
        save=True,
        sort=True
    )


if __name__ == '__main__':
    create_submission_dataset()