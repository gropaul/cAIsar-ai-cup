from typing import List
import numpy as np

def decimal_scaling_normalization(data_row: List[float]):

    data_row = np.array(data_row)

    max_value = int(np.round(np.max(np.abs(data_row))))
    digit=len(str(max_value))
    div=pow(10,digit)

    if div == 0: return data_row

    f_x = lambda x: np.round(x / div,decimals=8)

    data_row_normalized = f_x(data_row)
    return data_row_normalized