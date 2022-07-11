from typing import List
import numpy as np

from data_processing.normalization_metadata_setup import ABS_MEDIAN

required_data = [
    ABS_MEDIAN
]


def median_normalization(data_row: List[float], abs_median: float):
    
    data_row = np.array(data_row)

    if abs_median == 0: return 0

    f_x = lambda x: np.round(x / abs_median, decimals=8)

    data_row_normalized = f_x(data_row)
    return data_row_normalized