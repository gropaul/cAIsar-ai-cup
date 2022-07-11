from typing import List
import numpy as np
from data_processing.normalization_metadata_setup import MAX_ABS_VALUE

required_data = [
    MAX_ABS_VALUE
]

def decimal_scaling_normalization(data_row: List[float], max_abs_value: float):

    data_row = np.array(data_row)

    digit=len(str(max_abs_value))
    div=pow(10,digit)

    if div == 0: return data_row

    f_x = lambda x: np.round(x / div,decimals=8)

    data_row_normalized = f_x(data_row)
    return data_row_normalized