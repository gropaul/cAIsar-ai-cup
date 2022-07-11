
from importlib.resources import path
import pandas as pd
from utils.SYSCONFIG import DATA_SETS
import data_generator.batch_data_path_generator as pg
from utils.util_functions import parse_index, save_index, printc, CustomBar
import numpy as np
import json
import os

class NormalizationMetaData():

    def __init__(self, name:str, function) -> None:
        self.name = name
        self.function = function
        pass

AVERAGE = 'average'
MAX_ABS_VALUE = 'max_abs_value'
MIN_VALUE = 'min_value'
MAX_VALUE = 'max_value'
MEDIAN = 'median'
ABS_MEDIAN = 'abs_median'
STANDARD_DERIVATION = 'standard_derivation'

meta_data_vars = [
    NormalizationMetaData(
        name=AVERAGE, 
        function= lambda x : np.average(x)
    ),
    NormalizationMetaData(
        name=MAX_ABS_VALUE, 
        function= lambda x : int(np.round(np.max(np.abs(x))))
    ),
    NormalizationMetaData(
        name=MEDIAN, 
        function= lambda x : np.median(x)
    ),
    NormalizationMetaData(
        name=ABS_MEDIAN, 
        function= lambda x : np.median(np.abs(x))
    ),
    NormalizationMetaData(
        name=MIN_VALUE, 
        function= lambda x : np.min(x)
    ),
    NormalizationMetaData(
        name=MAX_VALUE, 
        function= lambda x : np.max(x)
    ),
    NormalizationMetaData(
        name=STANDARD_DERIVATION, 
        function= lambda x : np.std(x)
    ),
]


def create_normalization_metadata():

       
    def send_message(message: str, **kwargs) -> None:
        printc(source='[NormalizationMetadataSetup]', message=message, **kwargs)

    # create metadata_file for every file in dataset

    for dataset_name in DATA_SETS:
        original_data_index_path: str = pg.get_original_csv_index_path(
            data_subset_name = dataset_name
        )

        send_message(f'Starting to add meta data for {dataset_name}')

        index = parse_index(original_data_index_path, verify=True)

        paths = []

        bar =  CustomBar('Iterating files ... ', max=len(index))


        for file_path in index:

            bar.next()
            
            df = pd.read_csv(file_path)

            meta_data_map = {}


            for column_name in df:

                meta_data_map[column_name] = {}
                    
                for meta_data_var in meta_data_vars:

                    meta_name = meta_data_var.name
                  

                    name : str = column_name
                    values = df[column_name].tolist()

                    # Only take values that are full off numbers and are not the labels
                    if all([isinstance(value, str) for value in values]) or name in ['LFA','RFA']: 
                        # send_message(f'Skipping {name} as it does not contain only numbers or it is a label column')
                        continue

                    #send_message(f'Normalizing {name} ...')

                    value = meta_data_var.function(values)
                    meta_data_map[column_name][meta_name] = value

            file_name = os.path.basename(file_path)
            dir_name = os.path.dirname(file_path)
            meta_path = os.path.join(
                dir_name,
                file_name.replace('.csv', '_normalization_meta.json')
            )

            paths.append(meta_path)

            with open(meta_path, 'w', encoding='utf8') as json_file:
                json.dump(meta_data_map, json_file, ensure_ascii=False, indent=2)

        index_path = pg.get_original_norm_meta_index_path(data_subset_name=dataset_name)

        save_index(index_path=index_path, index=paths)
  
            

if __name__ == '__main__':
    create_normalization_metadata()