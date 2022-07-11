


from typing import List
import numpy as np

def sigmoid_normalization(data_row: List[float]):
    


    data_row = np.array(data_row)
    avg = np.average(data_row)
    std_derivation = np.std(data_row)


    f_x = lambda x: (x - avg) / std_derivation

    sigmoid = lambda x: 1/(1 + np.exp(-f_x(x)))

    fix_range = lambda x : np.round((x - 0.5) * 2,decimals=8)

    data_row_normalized = fix_range(sigmoid(data_row))
    return data_row_normalized