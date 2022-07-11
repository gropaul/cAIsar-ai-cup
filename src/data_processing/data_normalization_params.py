


class DataNormalizationParams:
    '''
    normalization_mode:
        'by_all': Normalization function is applied once on the whole dataset.
        'by_time_series': Normalization function is applied per time series. (-> once per file)
        'by_batch': !NOT IMPLEMENTED! Normalization function is applied per batch.

    normalization_function:
        For documentation, look at the implementations under data_processing/normalization functions
        NOTE: for no optimization, use 'original', this is only available with 'normalization_mode'='by_time_series'
    '''


    options = {
        'normalization_mode': ['by_time_series', 'by_all'], # default: 'by_time_series'
        'normalization_function' : [
            'original',
            'min_max_symmetrical',
            'tanh_estimator',
            'standardization',
            'median_normalization',
            'sigmoid_normalization',
            'decimal_scaling_normalization',
            'min_max_standardization',
        ] # default: 'original' 
    }

    # no changes at all
    default = {
        'normalization_function': 'original',
        'normalization_mode': 'by_time_series'
    }

    best4 = {
        'normalization_function': 'tanh_estimator',
        'normalization_mode': 'by_time_series'
    }

    decimal_scaling = {
        'normalization_function': 'decimal_scaling_normalization',
        'normalization_mode': 'by_time_series'
    }
    
    min_max = {
        'normalization_function': 'min_max_symmetrical',
        'normalization_mode': 'by_time_series'
    }