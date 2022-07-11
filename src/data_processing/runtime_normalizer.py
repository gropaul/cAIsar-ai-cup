

from data_processing.normalization_functions_runtime.min_max_symmetrical import min_max_symmetrical, required_data as min_max_required_data
from data_processing.normalization_functions_runtime.tanh_estimator import tanh_estimator, required_data as tanh_estimator_required_data
from data_processing.normalization_functions_runtime.standardization import standardization, required_data as standardization_required_data
from data_processing.normalization_functions_runtime.median_normalization import median_normalization, required_data as median_normalization_required_data
from data_processing.normalization_functions_runtime.sigmoid_normalization import sigmoid_normalization, required_data as sigmoid_normalization_required_data
from data_processing.normalization_functions_runtime.decimal_scaling_normalization import decimal_scaling_normalization, required_data as decimal_scaling_normalization_required_data
from data_processing.normalization_functions_runtime.min_max_standardization import min_max_standardization, required_data as min_max_standardization_required_data

from utils.util_functions import printc, get_partial_dict
import numpy as np

def send_message( message: str, **kwargs) -> None:
    printc(source='[RuntimeNormalizer]', message=message, **kwargs)
        
def apply_normalization(X, y, norm_meta_data, data_columns, normalization_function_name: str):


    n_columns = X.shape[1]
    for i in range(n_columns):
        column_name = data_columns[i]
        normalization_function = get_normalization_function(norm_meta_data, normalization_function_name, column_name)
        X[:,i] = normalization_function(X[:,i])

    return X, y


def get_normalization_function(norm_meta_data, normalization_function: str, column_name):

    try:
        column_meta_data = norm_meta_data[column_name]

        for key in column_meta_data.keys():
            column_meta_data[key] = np.array(column_meta_data[key])
    except:
        print(norm_meta_data)
        print(normalization_function)
        print(column_name)

   
    # Select a data from
    if normalization_function == 'original':
        norm_func = lambda row : row

    elif normalization_function == 'min_max_symmetrical':
        norm_func = lambda row : min_max_symmetrical(row,y_extent=1, **get_partial_dict(column_meta_data, min_max_required_data))

    elif normalization_function == 'tanh_estimator':
        map =  get_partial_dict(column_meta_data, tanh_estimator_required_data)
        norm_func = lambda row : tanh_estimator(row,**map)

    elif normalization_function == 'standardization':

        norm_func = lambda row : standardization(row, **get_partial_dict(column_meta_data, standardization_required_data))

    elif normalization_function == 'median_normalization':

        norm_func = lambda row : median_normalization(row, **get_partial_dict(column_meta_data, median_normalization_required_data))

    elif normalization_function == 'sigmoid_normalization':

        norm_func = lambda row : sigmoid_normalization(row, **get_partial_dict(column_meta_data, sigmoid_normalization_required_data))

    elif normalization_function == 'decimal_scaling_normalization':
        norm_func = lambda row : decimal_scaling_normalization(row, **get_partial_dict(column_meta_data, decimal_scaling_normalization_required_data))
    
    elif normalization_function == 'min_max_standardization':
        norm_func = lambda row : min_max_standardization(row, **get_partial_dict(column_meta_data, min_max_standardization_required_data))

    else:
        send_message(f'No normalization function was found for {normalization_function}! Using fallback min_max_symmetrical')
        norm_func = lambda row : min_max_symmetrical(row,y_extent=1, **get_partial_dict(column_meta_data, min_max_required_data))

    return norm_func