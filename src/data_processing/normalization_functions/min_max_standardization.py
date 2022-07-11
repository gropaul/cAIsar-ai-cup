from data_processing.normalization_functions.min_max_symmetrical import min_max_symmetrical
from data_processing.normalization_functions.standardization import standardization
from typing import List

# normalisiert Datenreihe, relativer Abstand zur Y-Achse wird beibehalten
# [-1,0,1,2] mit y_extent = 1 -> [-0.5, 0.0, 0.5, 1.0]

def min_max_standardization(data_row: List[float], y_extent: float = 1) -> List[float]:
    return standardization(min_max_symmetrical(data_row=data_row, y_extent=y_extent))