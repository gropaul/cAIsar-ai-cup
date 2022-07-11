




class DataAugmenterParams:

    '''
    new sets of parameters should be described as a set of changes made
    to the default parameter setting to illustrate the dependencies between
    different versions

    parameter versions available:
    default: default implementation for DataAugmenterParams
    '''
    
    default = {
        'seed': None,
    }

    best = {
        'seed': None,

        # Drift
        'drift_max': 0.11166,
        'drift_points': 50,
        'drift_kind': 'multiplicative',
        'drift_prob': 0.25,

        # Noise
        'noise_scale': 0.4,
        'noise_prob': 0.25,

        # Convolution
        'convolve_window_type': 'flattop',
        'convolve_window_size': 11,
        'convolve_prob': 0.25,

        # Dropout
        'dropout_percentage': 0.27,
        'dropout_prob': 0.27,
        'dropout_size': 4,
        'dropout_fill': 0.0,

        # TimeWarp
        'time_warp_changes': 29,
        'time_warp_max': 1.5,
        'time_warp_prob': 0.5,

    }

    best2 = {
        'seed': None,

        # Drift
        'drift_max': 0.11166,
        'drift_points': 50,
        'drift_kind': 'multiplicative',
        'drift_prob': 0.35,

        # Noise
        'noise_scale': 0.025,
        'noise_prob': 0.5,

        # Convolution
        'convolve_window_type': 'bohman',
        'convolve_window_size': 11,
        'convolve_prob': 0.35,

        # Dropout
        'dropout_percentage': 0.05,
        'dropout_prob': 0.65,
        'dropout_size': 1,
        'dropout_fill': 0.0,

        # TimeWarp
        'time_warp_changes': 13,
        'time_warp_max': 1.8,
        'time_warp_prob': 0.5,

    }

    # best config by 22.06.2022 - Created by using the config of the best run 
    # of the hyperoptimization
    best3 = {
        'seed': None,

        # Drift
        'drift_max': 0.16,
        'drift_points': 80,
        'drift_kind': 'multiplicative',
        'drift_prob': 0.1094742327980981,

        # Noise
        'noise_scale': 0.0195,
        'noise_prob': 0.66,

        # Convolution
        'convolve_window_type': 'flattop',
        'convolve_window_size': 9,
        'convolve_prob': 0.5,

        # Dropout
        'dropout_percentage': 0.1022201610567164,
        'dropout_prob': 0.62,
        'dropout_size': 1,
        'dropout_fill': 0.0,

        # TimeWarp
        'time_warp_changes': 42,
        'time_warp_max': 1.4,
        'time_warp_prob': 0.25,
    }

    # best config by 24.06.2022 - Created by looking at the avg config of the top 30 runs
    best4 = {
        'seed': None,

        # Drift
        'drift_max': 0.1,
        'drift_points': 58,
        'drift_kind': 'additive',
        'drift_prob': 0.6,

        # Noise
        'noise_scale': 0.045,
        'noise_prob': 0.4,

        # Convolution
        'convolve_window_type': 'bohman',
        'convolve_window_size': 11,
        'convolve_prob': 0.5,

        # Dropout
        'dropout_percentage': 0.22,
        'dropout_prob': 0.1,
        'dropout_size': 8,
        'dropout_fill': 0.0,

        # TimeWarp
        'time_warp_changes': 42,
        'time_warp_max': 1.4,
        'time_warp_prob': 0.85,
    }

    # best config by 10.07.2022 - Created by looking at the avg config of the top 30 runs
    best4 = {
        'seed': None,

        # Drift
        'drift_max': 0.1,
        'drift_points': 60,
        'drift_kind': 'multiplicative',
        'drift_prob': 0.8,

        # Noise
        'noise_scale': 0.05,
        'noise_prob': 0.5,

        # Convolution
        'convolve_window_type': 'flattop',
        'convolve_window_size': 8,
        'convolve_prob': 0.6,

        # Dropout
        'dropout_percentage': 0.3,
        'dropout_prob': 0.3,
        'dropout_size': 6,
        'dropout_fill': 'ffill',

        # TimeWarp
        'time_warp_changes': 20,
        'time_warp_max': 2,
        'time_warp_prob': 0.05,
    }