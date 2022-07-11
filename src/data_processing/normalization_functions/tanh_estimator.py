from typing import List
import numpy as np

def tanh_estimator(data_row: List[float]):
    
    data_row = np.array(data_row)
    mean = np.mean(data_row)
    std_derivation = np.std(data_row)

    f_x = lambda x: np.round(np.tanh(0.1 * (x - mean)/std_derivation), decimals=8)
    #f_x = lambda x: 0.5 *  (np.tanh(0.01 * (x-my)/sigmoid) + 1)

    data_row_normalized = f_x(data_row)
    return data_row_normalized