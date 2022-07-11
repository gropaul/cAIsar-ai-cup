from typing import List
import numpy as np
import pandas as pd
import os

import utils.util_functions as utils
from utils.SYSCONFIG import DATA_PATH, PLATFORM
from data_processing.normalization_functions.min_max_symmetrical import min_max_symmetrical
from data_processing.normalization_functions.tanh_estimator import tanh_estimator
from data_processing.normalization_functions.standardization import standardization
from data_processing.normalization_functions.median_normalization import median_normalization
from data_processing.normalization_functions.sigmoid_normalization import sigmoid_normalization
from data_processing.normalization_functions.decimal_scaling_normalization import decimal_scaling_normalization

import data_generator.batch_data_path_generator as pg

def send_message(message: str, **kwargs) -> None:
    utils.printc(source='[DATA NORMALIZATION]', message=message, **kwargs)

FILE_COLUMN_NAME = 'Path'


def load_csv_files(path_list: List[str]) -> List[pd.DataFrame]:
    dfs : List[pd.DataFrame] = []

    file_count = len(path_list)
    bar =  utils.CustomBar('Loading files ... ', max=file_count)

    index = 0

    for path in path_list:
        #if index > 5: break
        index += 1
        df : pd.DataFrame = pd.read_csv(path, index_col=False)
        df[FILE_COLUMN_NAME] = path
        dfs.append(df)
        bar.next()

    return dfs

def normalize_df(df: pd.DataFrame, normalization_function) -> pd.DataFrame:
    # get iterate though each row

    for column_name in df:

        name : str = column_name
        values = df[column_name].tolist()

        # Only take values that are full off numbers
        if all([isinstance(value, str) for value in values]) or name in ['LFA','RFA']: 
            # send_message(f'Skipping {name} as it does not contain only numbers or it is a label column')
            continue

        #send_message(f'Normalizing {name} ...')

        normalized_values = normalization_function(values)
        df[column_name] = normalized_values
    
    return df

def split_and_save_df(
    df: pd.DataFrame,
    original_file_index: List[str],
    target_folder_path: str,
    target_index_path: str
) -> List[pd.DataFrame]: 
    
    paths_count = len(original_file_index)
    bar = utils.CustomBar('Saving DataFrames to files ... ', max=paths_count)

    paths : List[str] = []

    utils.create_folder_if_not_exists(target_folder_path)

    delimiter = '\\' if PLATFORM == 'WINDOWS' else '/'

    for path in original_file_index:
        
        frame = df.loc[(df[FILE_COLUMN_NAME] == path)]
        frame = frame.drop(columns=[FILE_COLUMN_NAME])

        file_name = path.split(delimiter)[-1]
        new_path = os.path.join(target_folder_path,file_name)

        paths.append(new_path)
        frame.to_csv(new_path,index=False)

        header = str(list(frame.columns)).replace('[','').replace(']','').replace("'","").replace(', ',',')

        np.savetxt(new_path, frame.values, delimiter=",",fmt='%10.8f',header=header,comments='')

        bar.next()

    with open(target_index_path, 'w') as f:
        for path in paths:
            f.write('%s\n' % path)

def data_concatenation(normalization_mode: str, data_frames: List[pd.DataFrame]) ->List[pd.DataFrame]: 
    '''_summary_

    Args:
        normalization_mode (str): The mode of the normalization, which defines in 
        which subsets the files are passed to the normalization function
        data_frames (List[pd.DataFrame]): The data as one DataFrame per file/time series 

    Returns:
        List[pd.DataFrame]: The data separated accordingly to the normalization_mode
    '''

    send_message(f'Started concatenation of the data with the following mode: {normalization_mode}')


    # combine the whole data to one big DataFrame
    if normalization_mode == 'by_all':
        combined_frame = pd.concat(data_frames)
        return [combined_frame]
    
    # return the data with one DataFrame by file/time series
    if normalization_mode == 'by_time_series':
        return data_frames

    send_message(f'No normalization_mode found for {normalization_mode}')
    return None

def apply_normalization(data_frames: List[pd.DataFrame], normalization_function: str) -> List[pd.DataFrame]:

    dfs_count = len(data_frames)
    bar =  utils.CustomBar('Normalizing data frames ... ', max=dfs_count)


    send_message(f'Starting normalization of {dfs_count} DataFrames using {normalization_function}')

    # define a normalization function for the data frame
    norm_func = None

    # Select a data from
    if normalization_function == 'min_max_symmetrical':
        norm_func = lambda row : min_max_symmetrical(row,y_extent=1)

    elif normalization_function == 'tanh_estimator':
        norm_func = lambda row : tanh_estimator(row)

    elif normalization_function == 'standardization':
        norm_func = lambda row : standardization(row)

    elif normalization_function == 'median_normalization':
        norm_func = lambda row : median_normalization(row)

    elif normalization_function == 'sigmoid_normalization':
        norm_func = lambda row : sigmoid_normalization(row)

    elif normalization_function == 'decimal_scaling_normalization':
        norm_func = lambda row : decimal_scaling_normalization(row)

    else:
        send_message('No normalization function was found! Using fallback min_max_symmetrical')
        norm_func = lambda row : min_max_symmetrical(row,y_extent=1)

    normalized_data_frames = [None] * dfs_count 

    # iterate though all data_frames and normalize them
    for i in range(dfs_count):
        data_frame = data_frames[i]
        normalized_data_frame = normalize_df(data_frame,norm_func)
        normalized_data_frames[i] = normalized_data_frame

        bar.next()

    return normalized_data_frames




def normalize(
    dataset_name: str,
    normalization_mode: str,
    normalization_function: str,
):

    send_message('Started normalization for the following parameters:')
    send_message(f'dataset_name: {dataset_name}')
    send_message(f'normalization_mode: {normalization_mode}')
    send_message(f'normalization_function: {normalization_function}')


    original_data_index_path: str = pg.get_original_csv_index_path(
        data_subset_name = dataset_name
    )

    # load the index of the original data
    original_data_index : List[str] = utils.parse_index(original_data_index_path, verify=True)

    # load the data
    original_data : List[pd.DataFrame] = load_csv_files(original_data_index)

    separated_data = data_concatenation(normalization_mode=normalization_mode, data_frames=original_data)

    normalized_data = apply_normalization(data_frames=separated_data,normalization_function=normalization_function)

    # create on combined big table to reuse saving code
    combined_normalized_data = pd.concat(normalized_data)

    # split the previously merged CSVs and save them with the corresponding appendix

    normalization_folder_path: str = pg.get_folder_path(
        data_subset_name = dataset_name,
        normalization_function = normalization_function,
        normalization_mode = normalization_mode,
    )

    normalized_data_index_path: str = pg.get_csv_index_path(
        data_subset_name = dataset_name,
        normalization_function = normalization_function,
        normalization_mode = normalization_mode,
    )
    
    split_and_save_df(
        df = combined_normalized_data,
        original_file_index = original_data_index,
        target_folder_path = normalization_folder_path,
        target_index_path = normalized_data_index_path
    )
