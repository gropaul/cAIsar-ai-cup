from datetime import datetime
import json
from typing import List,Dict,Any
import pandas as pd
from progress.bar import Bar
from os.path import exists
import numpy as np
from utils.SYSCONFIG import DATA_PATH, PLATFORM
import os
from utils.errors import DataFilesNotFoundError

'''
contains various utility functions that are useful across files and modules
'''

def parse_index(index_file: str, verify: bool = False) -> List[str]:
    '''
    returns the value of the index generated for 
    the data dir by indexing all *.csv/*.json files using 
    a batch file
    '''

    with open(index_file) as f:
        contents = f.read()
    
    index = contents.split('\n')
    index = [entry for entry in index if entry != '']
    
    index_cropped_names = [entry[:-1] for entry in index if entry[-1:] == ' ']
    index_non_cropped_names = [entry for entry in index if entry[-1:] != ' ']
    index = [*index_cropped_names, *index_non_cropped_names]

    if verify:
        missing_files: List[str] = []
        for entry in index:
            if not(exists(entry)):
                missing_files.append(entry)
        if len(missing_files) > 0:
            raise DataFilesNotFoundError(files=missing_files)

    return index


def save_index(index_path:str,index:List[str]):
    """Saves an fileindex to a file

    Args:
        index_path (str): The path to save the index to
        index (List[str]): The index itself, a list of file paths 
    """
    with open( index_path, 'w') as f:
        for path in index:
            f.write("%s\n" % path)


def printc(source: str = '[Unknown]', message: str = '', leading: str = '', **kwargs) -> None:
    '''
    prints [message] with a [src] and a timestamp attached
    '''
    print(f'{leading}{datetime.now()}   {source} {message}')


def create_zero_frame():
    """Generates a data frame filled with zeros in every column of length 4096
    """

    zero_dict = {}
    for col in ['LAV', 'LAX', 'LAY', 'LAZ', 'LRV', 'LRX', 'LRY', 'LRZ', 'RAV', 'RAX',
           'RAY', 'RAZ', 'RRV', 'RRX', 'RRY', 'RRZ', 'LFA', 'RFA']:
        zero_dict[col] = np.zeros(4096)
    zero_df = pd.DataFrame(zero_dict)
    zero_path = os.path.join(DATA_PATH, 'zero.xsv')
    zero_df.to_csv(zero_path, index=False)


def create_zero_meta_data():
    zero_dict = {
        'Subject' : 0,
        'Trial' : 0,
        'Code' : '0-0',
        'Age': 'NC', 
        'Gender': 'NC', 
        'Height': 'NC', 
        'Weight': 'NC', 
        'BMI': 'NC', 
        'Laterality': 'NC', 
        'Sensor': 'NC', 
        'WalkedDistance': 'NC', 
        'WalkingSpeed': 'NC', 
        'PathologyGroup': 'NC', 
        'IsControl': 'NC',
        'LeftFootActivity' : [],
        'RightFootActivity' : [],
        }
    
    zero_path = os.path.join(DATA_PATH, 'zero.xson')
    with open(zero_path, 'w') as f:
        json.dump(zero_dict, f)    


def create_zero_norm_meta_data():
    zero_dict = {
        "LAV": {
            "average": 0,
            "max_abs_value": 0,
            "median": 0,
            "abs_median": 0,
            "min_value": 0,
            "max_value": 0,
            "standard_derivation": 0
        },
        "LAX": {
            "average": 0,
            "max_abs_value": 0,
            "median": 0,
            "abs_median": 0,
            "min_value": 0,
            "max_value": 0,
            "standard_derivation": 0
        },
        "LAY": {
            "average": 0,
            "max_abs_value": 0,
            "median": 0,
            "abs_median": 0,
            "min_value": 0,
            "max_value": 0,
            "standard_derivation": 0
        },
        "LAZ": {
            "average": 0,
            "max_abs_value": 0,
            "median": 0,
            "abs_median": 0,
            "min_value": 0,
            "max_value": 0,
            "standard_derivation": 0
        },
        "LRV": {
            "average": 0,
            "max_abs_value": 0,
            "median": 0,
            "abs_median": 0,
            "min_value": 0,
            "max_value": 0,
            "standard_derivation": 0
        },
        "LRX": {
            "average": 0,
            "max_abs_value": 0,
            "median": 0,
            "abs_median": 0,
            "min_value": 0,
            "max_value": 0,
            "standard_derivation": 0
        },
        "LRY": {
            "average": 0,
            "max_abs_value": 0,
            "median": 0,
            "abs_median": 0,
            "min_value": 0,
            "max_value": 0,
            "standard_derivation": 0
        },
        "LRZ": {
            "average": 0,
            "max_abs_value": 0,
            "median": 0,
            "abs_median": 0,
            "min_value": 0,
            "max_value": 0,
            "standard_derivation": 0
        },
        "RAV": {
            "average": 0,
            "max_abs_value": 0,
            "median": 0,
            "abs_median": 0,
            "min_value": 0,
            "max_value": 0,
            "standard_derivation": 0
        },
        "RAX": {
            "average": 0,
            "max_abs_value": 0,
            "median": 0,
            "abs_median": 0,
            "min_value": 0,
            "max_value": 0,
            "standard_derivation": 0
        },
        "RAY": {
            "average": 0,
            "max_abs_value": 0,
            "median": 0,
            "abs_median": 0,
            "min_value": 0,
            "max_value": 0,
            "standard_derivation": 0
        },
        "RAZ": {
            "average": 0,
            "max_abs_value": 0,
            "median": 0,
            "abs_median": 0,
            "min_value": 0,
            "max_value": 0,
            "standard_derivation": 0
        },
        "RRV": {
            "average": 0,
            "max_abs_value": 0,
            "median": 0,
            "abs_median": 0,
            "min_value": 0,
            "max_value": 0,
            "standard_derivation": 0
        },
        "RRX": {
            "average": 0,
            "max_abs_value": 0,
            "median": 0,
            "abs_median": 0,
            "min_value": 0,
            "max_value": 0,
            "standard_derivation": 0
        },
        "RRY": {
            "average": 0,
            "max_abs_value": 0,
            "median": 0,
            "abs_median": 0,
            "min_value": 0,
            "max_value": 0,
            "standard_derivation": 0
        },
        "RRZ": {
            "average": 0,
            "max_abs_value": 0,
            "median": 0,
            "abs_median": 0,
            "min_value": 0,
            "max_value": 0,
            "standard_derivation": 0
        },
        "LFA": {},
        "RFA": {}
    }
    
    zero_path = os.path.join(DATA_PATH, 'zero_norm.xson')
    with open(zero_path, 'w') as f:
        json.dump(zero_dict, f)    


class CustomBar(Bar):
    space_str = ' ' * 10
    suffix = '%(index)d/%(max)d - ETA: %(eta_minutes)d min' + space_str
    
    @property
    def eta_minutes(self):
        return self.eta // 60

def shift_array(arr, num, fill=np.nan):
    # https://stackoverflow.com/questions/30399534/shift-elements-in-a-numpy-array#30534478
    # cf. shift5
    
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result

def convert_mask_to_cup_format(arr) -> list:
    # create shifted array with orignal[n+1] = shifted[n]
    shifted_arr = shift_array(arr, -1, fill=arr[-1])
    
    # step_changes must differ with in value with their successor
    # begin of step arr[n] = 0, arr[n+1] = 1
    # -> step starts at n+1
    # end of step arr[n] = 1, arr[n+1] = 0
    # -> step ends at n
    step_changes = np.logical_xor(arr, shifted_arr)
    
    # create array with indices and apply boolean mask for selection
    changes = np.arange(0, len(arr))[step_changes]
    # add start of first step at 0 if the prediction starts with a step
    if arr[0] == 1:
        changes = np.insert(changes, 0, -1)
    # add end of last step at len(arr) - 1 if the prediction ends with a step
    if arr[-1] == 1:
        changes = np.insert(changes, len(changes), len(arr) - 1)
    
    # explanation: see comments above and *.ipynb
    correct_starts = np.tile([1, 0], int(len(changes)/2))
    changes += correct_starts
    
    # convert array (vector) to matrix
    nested = changes.reshape(int(len(changes) / 2), 2)
    # delete all steps of length one, i. e. start == end
    nested = nested[~(nested[:, 0] == nested[:, 1])]
    nested = nested.tolist()
    return nested


def convert_float_to_binary_mask(mask: np.array, threshold: float = 0.5):
    # NOTE: the brackets need to remain to mark this operation
    # as assignment, leading to the bool-to-float-conversion
    # False -> 0.0, True -> 1.0
    mask[:] = mask[:] >= threshold
    return mask

def get_updated(params : Dict[str, Any], **kwargs) -> Dict[str, Any]:
    params = params.copy()
    params.update(**kwargs)
    return params

def get_partial_dict(dictionary: dict, keys: list):
    partial_dict = {}
    for key in keys:
        partial_dict[key] = dictionary[key]
    return partial_dict

def send_message(message: str, **kwargs) -> None:
    printc(source='[UtilFunctions]', message=message, **kwargs)

def create_folder_if_not_exists(path):
    
    folder_exists = os.path.exists(path)

    if not folder_exists:
        # Create a new directory because it does not exist 
        os.makedirs(path)
        send_message(f"A new folder was created under {path}.")
    else:
        send_message(f"Found an existing folder for {path}")

def get_folder_size(path):
    return sum(os.path.getsize(path + "//" + f) for f in os.listdir(path) if os.path.isfile(path + "//" +f))


def transform_predictions_to_ts(y_pred):
    def process_channel(mask: np.array) -> list:
        # process one mask of shape (length,)
        binary_mask = convert_float_to_binary_mask(mask)
        steps = convert_mask_to_cup_format(binary_mask)
        steps = np.array(steps).tolist()
        return steps

    predictions = []

    for mask_pred in y_pred:
        multi_channel: bool = (len(y_pred[0].shape) > 1)
        channels = y_pred[0].shape[1] if multi_channel else 1

        for channel in range(channels):
            channel_pred = mask_pred[:, channel] if multi_channel else mask_pred
            processed_pred = process_channel(mask=channel_pred)
            predictions.append(processed_pred)

    return predictions


def reduce_list(list_to_remove, percentage):
    """Reduced a list to percentage: 10 elements reduced to 0.2 -> 2 elements

    Args:
        list_to_remove (_type_): The list to be removed
        percentage (_type_): the percentage by witch the list should be reduced to: 

    Returns:
        _type_: A reduced list
    """
    length = len(list_to_remove)
    n_removed = 0
    n_processed = 0
    new_list = []

    for i in range(length):
        current_percentage_removed = 0 if n_processed == 0 else n_removed / n_processed
        if current_percentage_removed <= 1 - percentage and percentage != 1.0 :
            n_removed += 1
        else: 
            new_list.append(list_to_remove[i])

        n_processed += 1

    return new_list