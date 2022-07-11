
# Standardization
#It is one of the most common standardization technique. You find the z-scores of your variables on their own distribution.
# Si=(Xi−mean(Xi))∗std(Xi)


from typing import List
import numpy as np

def standardization(data_row: List[float]):
    
    data_row = np.array(data_row)
    avg = np.average(data_row)
    std_derivation = np.std(data_row)

    f_x = lambda x: np.round((x - avg) / std_derivation,decimals=8)

    data_row_normalized = f_x(data_row)
    return data_row_normalized