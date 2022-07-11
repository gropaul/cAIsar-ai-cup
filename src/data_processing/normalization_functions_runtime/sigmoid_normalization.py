


from typing import List
import numpy as np

from data_processing.normalization_metadata_setup import AVERAGE, STANDARD_DERIVATION

required_data = [
    AVERAGE, STANDARD_DERIVATION
]


def sigmoid_normalization(data_row: List[float], average: float, standard_derivation: float,  ):
    
    data_row = np.array(data_row)

    if standard_derivation == 0: return data_row

    f_x = lambda x: (x - average) / standard_derivation

    sigmoid = lambda x: 1/(1 + np.exp(-f_x(x)))

    fix_range = lambda x : np.round((x - 0.5) * 2,decimals=8)

    data_row_normalized = fix_range(sigmoid(data_row))
    return data_row_normalized