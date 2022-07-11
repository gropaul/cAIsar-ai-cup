from data_processing.normalization_functions_runtime.min_max_symmetrical import min_max_symmetrical
from data_processing.normalization_functions_runtime.standardization import standardization
from typing import List

from numpy import average

from data_processing.normalization_metadata_setup import AVERAGE, MAX_VALUE, MIN_VALUE, STANDARD_DERIVATION

required_data = [
    AVERAGE, STANDARD_DERIVATION, MIN_VALUE, MAX_VALUE
]


def min_max_standardization(
    data_row: List[float], 
    min_value: float, max_value: float, 
    average: float, standard_derivation: float,
    y_extent: float = 1
) -> List[float]:

    stand_res = standardization(data_row, average=average, standard_derivation=standard_derivation)
    min_max_res = min_max_symmetrical(
        data_row=stand_res, y_extent=y_extent,
        max_value=max_value, min_value=min_value
    )
    return min_max_res