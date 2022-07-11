from typing import List
import numpy as np

# normalisiert Datenreihe, relativer Abstand zur Y-Achse wird beibehalten
# [-1,0,1,2] mit y_extent = 1 -> [-0.5, 0.0, 0.5, 1.0]


from data_processing.normalization_metadata_setup import MIN_VALUE, MAX_VALUE

required_data = [
    MIN_VALUE, MAX_VALUE
]


def min_max_symmetrical(data_row: List[float], min_value: float, max_value: float, y_extent: float = 1.0,) -> List[float]:

    data_row = np.array(data_row)

    max_y_extent = np.max([np.abs(min_value),np.abs(max_value)])

    range = max_y_extent * 2
    normalized_range = y_extent * 2

    if range == 0: return data_row

    normalization_factor = normalized_range / range

    f_x = lambda x: np.round(x * normalization_factor, decimals=8)

    data_row_normalized = f_x(data_row)
    return data_row_normalized