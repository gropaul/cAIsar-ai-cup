from typing import List
import numpy as np

from data_processing.normalization_metadata_setup import AVERAGE, STANDARD_DERIVATION

required_data = [
    AVERAGE, STANDARD_DERIVATION
]

def tanh_estimator(data_row: List[float], average: float, standard_derivation: float,):
    
    data_row = np.array(data_row)

    if standard_derivation == 0: return data_row

    f_x = lambda x: np.round(np.tanh(0.1 * (x - average)/standard_derivation), decimals=8)

    data_row_normalized = f_x(data_row)
    return data_row_normalized