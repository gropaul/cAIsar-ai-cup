from typing import Any, Dict
import utils.util_functions as utils

class WenKeyesParams:
    '''
    meaning of the parameters explained in the default param set
    new sets of parameters should be described as a set of changes made
    to the default parameter setting to illustrate the dependencies between
    different versions

    parameter versions available:
    default: default implementation as presented in the Wen and Keyes paper
    '''
    
    default = {
        # TODO: edit these manually and accordingly in the respective versions
        # basic dimensions of the time-series input encoded as
        # INPUT_SHAPE -> (length x channels) = (SEQUENCE_LENGTH, CHANNELS)
        'sequence_length' : 1024, 
        'channels' : 16,
        
        # NOTE: params inherent to the architecture and the problem
        'concat_axis' : 2,
        'base_filter_count' : 16, 
        'block_count' : 9, 
        
        # number of anomaly classes encoded as CLASS_COUNT; might exceed 1
        # as e. g. seasonal, time-independent etc. anomalies are differentiated
        'class_count' : 2, 

        # NOTE: params defined by Wen and Keyes in their paper
        # see paper for meaning
        'kernel_size' : {'default' : 3}, 
        'pool_size' : {'default' : 4},
        'dropout' : {'default' : 0.0},
        'strides' : {'default' : 1}
    }


    v1 = default.copy()
    v1.update({
        'base_filter_count' : 32,  
    })