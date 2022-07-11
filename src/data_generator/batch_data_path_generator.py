import os
from requests import patch
from utils.SYSCONFIG import DATA_GAIT, DATA_PATH, DATA_SUBSET_CONFIG,PROJECT_PATH
from utils.util_functions import create_folder_if_not_exists
data_subset_name = 1


def get_original_csv_index_path(data_subset_name: str) -> str:
    return get_csv_index_path(
        data_subset_name=data_subset_name
    )


def get_original_json_index_path(data_subset_name: str) -> str:
    return get_json_index_path(
        data_subset_name=data_subset_name
    )


def get_original_norm_meta_index_path(data_subset_name: str) -> str:
    return get_norm_meta_index_path(
        data_subset_name=data_subset_name
    )


def get_original_folder_path(data_subset_name: str) -> str:
    return get_folder_path(
        data_subset_name=data_subset_name
    )


def get_original_table_name(data_subset_name: str) -> str:
    return get_table_name(
        data_subset_name=data_subset_name
    )



def get_table_name(data_subset_name: str) -> str:

    data_folder_name = get_data_name(data_subset_name)

    folderPath = os.path.join(
        PROJECT_PATH,"tables",
        data_folder_name,
    )
    
    create_folder_if_not_exists(folderPath)
    return os.path.join(folderPath, data_subset_name + f'_table.pkl')



def get_meta_table_name(data_subset_name: str) -> str:

    data_folder_name = get_data_name(data_subset_name)

    folderPath = os.path.join(
        PROJECT_PATH,"metadata",
        data_folder_name,
        )
    create_folder_if_not_exists(folderPath)
        
    return os.path.join(folderPath, data_subset_name + f'_meta.json')


def get_csv_index_path(
    data_subset_name: str
):
    data_name = get_data_name(data_subset_name)

    return os.path.join(
        DATA_PATH,
        data_name,
        'csv_index_' + data_subset_name + '.txt'
    )


def get_json_index_path(data_subset_name: str):
    data_name = get_data_name(data_subset_name)
    return os.path.join(
        DATA_PATH,
        data_name,
        'json_index_' + data_subset_name + '.txt'
    )

def get_norm_meta_index_path(data_subset_name: str):
    data_name = get_data_name(data_subset_name)
    return os.path.join(
        DATA_PATH,
        data_name,
        'normalization_meta_index_' + data_subset_name + '.txt'
    )


def get_folder_path(data_subset_name: str):
    data_name = get_data_name(data_subset_name)
    path = os.path.join(
        DATA_PATH, 
        data_name
    )
    create_folder_if_not_exists(path)
    return path


# Little HACK: To create a random index in validation, it is necessary to 
# add a fallback data_set if the newly randomly created index is not registerable 
# in the DATA_SUBSET_CONFIG

def get_data_name(data_subset_name: str):

    if data_subset_name in DATA_SUBSET_CONFIG:
        data_subset_config = DATA_SUBSET_CONFIG[data_subset_name]
        data_name = data_subset_config['data']

        return data_name
    
    else: 
        return DATA_GAIT
