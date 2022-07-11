
from utils.util_functions import parse_index, save_index
from utils.SYSCONFIG import DATA_GAIT
from data_generator.batch_data_path_generator import get_csv_index_path
from data_generator.batch_generator_params import BatchGeneratorParams
from typing import List, Tuple
import random


TRAIN_TEST_SPLIT = 0.2
VALIDATION_RUN_COUNT = 5

def create_random_batch_generator_configs(session_config) -> List[Tuple[BatchGeneratorParams,BatchGeneratorParams]]:

    bg_train_config = session_config['bg_train_config']
    bg_val_config = session_config['bg_evaluation_config']
    normalization_settings = bg_train_config['normalization_settings']
    normalization_function = normalization_settings['normalization_function']
    normalization_mode = normalization_settings['normalization_mode']

    whole_dataset_index_path = get_csv_index_path(
        data_subset_name=DATA_GAIT,
        normalization_function=normalization_function,
        normalization_mode=normalization_mode
    )

    new_configs = []

    for i in range(VALIDATION_RUN_COUNT): 
        whole_dataset_index : List[str] = parse_index(index_file=whole_dataset_index_path, verify=True)
        number_of_validation_files = round( len(whole_dataset_index) * TRAIN_TEST_SPLIT)
    
        val_index = random.sample(whole_dataset_index,number_of_validation_files)
        train_index = [x for x in whole_dataset_index if x not in val_index]

        train_index_name = f'session_validation_random_train_{i}'
        val_index_name = f'session_validation_random_val_{i}'

        train_index_path = get_csv_index_path(
            data_subset_name=train_index_name,
            normalization_function=normalization_function,
            normalization_mode=normalization_mode
        )

        val_index_path = get_csv_index_path(
            data_subset_name=val_index_name,
            normalization_function=normalization_function,
            normalization_mode=normalization_mode
        )

        save_index(index=train_index, index_path=train_index_path)
        save_index(index=val_index, index_path=val_index_path)

        new_bg_train_config = bg_train_config.copy()
        new_bg_train_config.update({
            'data_subset_name' : train_index_name,
        })

        new_bg_val_config = bg_val_config.copy()
        new_bg_val_config.update({
            'data_subset_name' : val_index_name,
        })

        new_configs.append(
            (new_bg_train_config, new_bg_val_config)
        )

    return new_configs