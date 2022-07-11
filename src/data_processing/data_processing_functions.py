from typing import Dict, List
import pandas as pd
import json

import utils.util_functions as utils
from utils.errors import DataFilesNotFoundError
from utils.SYSCONFIG import PLATFORM

def add_annotations(df: pd.DataFrame, meta: dict, col_names:dict = {'LeftFootActivity' : 'LFA', 'RightFootActivity' : 'RFA'}) -> pd.DataFrame:
    df = df.copy(deep=True)
    
    for key in col_names.keys():
        annotations = meta[key]
        df[col_names[key]] = 0
        for annotation in annotations:
            df.loc[(df.index >= annotation[0]) & (df.index <= annotation[1]), [col_names[key]]] = 1
    
    return df

def match_csv_normalization_data_pairs(csv_index_file: str, normalization_data_index_file: str):
    csv_index = utils.parse_index(csv_index_file, verify=True)
    norm_index = utils.parse_index(normalization_data_index_file, verify=True)
    pairs = {}

    missing_files: List[str] = []

    delimiter = '\\' if PLATFORM == 'WINDOWS' else '/'

    for entry in csv_index:
        pair = {} 
        pair['csv'] = entry
        key = (entry.split(delimiter)[-1])[:-4]
        
        for possible_match in norm_index:
            if (delimiter + key + '_normalization_meta.json') in possible_match:
                pair['json'] = possible_match
                norm_index.remove(possible_match)
                break
        
        try:
            pair['json']
            pairs[key] = pair
        except KeyError:
            missing_files.append(f'{entry} -> corresponding meta*.json')
    
    if len(norm_index) > 0:
        for json_file in norm_index:
            missing_file = (f'{json_file} -> corresponding *.csv')
            missing_files.append(missing_file)
    if len(missing_files) > 0:
        raise DataFilesNotFoundError(files = missing_files)

    return pairs




def match_csv_json_pairs(csv_index_file: str, json_index_file: str) -> Dict[str, Dict[str, str]]:
    csv_index = utils.parse_index(csv_index_file, verify=True)
    json_index = utils.parse_index(json_index_file, verify=True)
    pairs = {}
    
    missing_files: List[str] = []

    delimiter = '\\' if PLATFORM == 'WINDOWS' else '/'
    
    for entry in csv_index:
        pair = {} 
        pair['csv'] = entry
        key = (entry.split(delimiter)[-1])[:-4]
        
        for possible_match in json_index:
            if (delimiter + key + '.json') in possible_match:
                pair['json'] = possible_match
                json_index.remove(possible_match)
                break
        
        try:
            pair['json']
            pairs[key] = pair
        except KeyError:
            missing_files.append(f'{entry} -> corresponding *.json')
    
    if len(json_index) > 0:
        for json_file in json_index:
            missing_file = (f'{json_file} -> corresponding *.csv')
            missing_files.append(missing_file)
    if len(missing_files) > 0:
        raise DataFilesNotFoundError(files = missing_files)
        
    return pairs

def annotate_all_files(csv_index_file: str, json_index_file: str, replace: bool = True, output_dir: str = None):
    file_pairs = match_csv_json_pairs(csv_index_file, json_index_file)
    bar =  utils.CustomBar('Annotating files', max=len(file_pairs.keys()))

    for key in file_pairs.keys():
        pair: dict = file_pairs[key]
        
        df: pd.DataFrame = pd.read_csv(pair['csv'], index_col=False)
        with open(pair['json']) as f:
            meta: dict = json.load(f)

        annotated_df = add_annotations(df, meta)

        if replace:
            annotated_df.to_csv(pair['csv'], index=False)
        else:
            annotated_df.to_csv(output_dir + key + '.csv', index=False)
        bar.next()
