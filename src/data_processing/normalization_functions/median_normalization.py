from typing import List
import numpy as np

def median_normalization(data_row: List[float]):
    
    data_row = np.array(data_row)
    median = np.median(np.abs(data_row))

    if median == 0: 
        return data_row

    f_x = lambda x: np.round(x / median, decimals=8)

    data_row_normalized = f_x(data_row)
    return data_row_normalized