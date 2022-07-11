import imp
from typing import Tuple, List
from utils.util_functions import printc
import re
import os

def send_message(message: str, **kwargs) -> None:
    printc(source='[DATA SETUP WORKFLOW]', message=message, **kwargs)


# endregion
def get_index_from_path(path: str) -> Tuple[int,int]:
    pattern = "[0-9]{1,}-[0-9]{1,}"
    x = re.findall(pattern,path)
    if len(x) != 1: send_message(f"ERROR WHILE PARSING PATH: got {x} extracted from {path}, there should only be one")

    first_index, second_index = x[0].split("-")

    return int(first_index), int(second_index)

def get_key_of_path(path1: str) -> int:

    path1_i1, path1_i2 = get_index_from_path(path1)

    # print(f"{path1}: {path1_i1},{path1_i2}")

    return path1_i1 * 1000 + path1_i2


def sort_index(index: List[str]) -> List[str]:
    index.sort(key=get_key_of_path)
    return index


def create_index_for_dir(dir_path, index_path, file_ending = None,save=True,sort=True) -> List[str]:

    index = []

    for file in os.listdir(dir_path):
        if file_ending is not None:
            if file.endswith(file_ending):
                index.append(os.path.join(dir_path,file))
        else:
            index.append(os.path.join(dir_path,file))

    if sort: 
        sort_index(index)

    if save:
        with open( index_path, 'w') as f:
            for path in index:
                f.write("%s\n" % path)

    return index
