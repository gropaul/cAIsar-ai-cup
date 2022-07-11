from typing import Any, Dict
from xml.dom.pulldom import default_bufsize
from data_processing.data_normalization_params import DataNormalizationParams
import numpy as np


from utils.SYSCONFIG import DATA_GAIT, TABLES_PATH, DATA_SUBSET_GAIT_TRAINING, DATA_SUBSET_GAIT_VALIDATION, DATA_SUBMISSION

from utils.SYSCONFIG import DATA_PATH
from data_generator.data_augmenter import DataAugmenter

class BatchGeneratorParams:

    '''
    new sets of parameters should be described as a set of changes made
    to the default parameter setting to illustrate the dependencies between
    different versions

    parameter versions available:
    default: default implementation for BatchGenerator
    '''
    
    default = {
        'normalization_settings': DataNormalizationParams.default,
        'data_subset_name': DATA_GAIT,
        # data_columns is a list of all columns that are part of the X variable
        'data_columns' : ['LAV', 'LAX', 'LAY', 'LAZ', 
                        'LRV', 'LRX', 'LRY', 'LRZ', 
                        'RAV', 'RAX', 'RAY', 'RAZ', 
                        'RRV', 'RRX', 'RRY', 'RRZ'],
        # label_columns is a list of all columns that are part of the y variable
        'label_columns' : ['LFA', 'RFA'],
        # the batch_size and shuffle setting; see Batches on Notion
        'batch_size' : 32,
        'shuffle' : True,
        # providing a data type for all columns prevents pandas from inferring the data type
        # itself, leading to possibly unexpected behavior
        'dtype' : {
                 'LAV': np.float64, 'LAX': np.float64, 'LAY': np.float64,
                 'LAZ': np.float64, 'LRV': np.float64, 'LRX': np.float64, 
                 'LRY': np.float64, 'LRZ': np.float64, 'RAV': np.float64, 
                 'RAX': np.float64, 'RAY': np.float64, 'RAZ': np.float64, 
                 'RRV': np.float64, 'RRX': np.float64, 'RRY': np.float64, 
                 'RRZ': np.float64, 'LFA': np.int8,    'RFA': np.int8
                 },
                 
        # controls the verbosity of the output
        'verbose' : True,

        # length of one sample along the time-axis 
        'length' : 1024,
        # type of generator; inverts the meaning of the train_test_split, prevents augmentation 
        'validation_generator' : False,
        # leads to a split of the data along the time-axis if set to a different value 0.0 < x < 1.0
        # the first x (percent) only may be used to draw samples
        'train_test_split' : 1.0,
        # determines the sampling frequency; if sample_freq < length the generated samples overlap
        'sample_freq' : 512,
        # determines the sampling offset; if sample_offset > 0 that part of each time-series is ignored
        'sample_offset' : 0,
        # whether to load files once and store them in-memory; only relevant for data sets above a certain size (~10 GB)
        'caching' : False,
        # whether to pad the time series with zeros, to ensure the entire time-series is sampled
        'padding' : False,
        # whether to pad the batch with zero samples, to ensure all samples are used
        'pad_last_batch': False,
        # should meta data be included, defaults to False
        'meta': False

    }


    augmented_v1 = default.copy()
    augmented_v1.update(
        {
            'augmenter':  DataAugmenter.v1,
        }
    )

    stepped_v1 = default.copy()

    stepped_v1.update(
        {
            'index_file' :  f'{DATA_PATH}\\csv_index-step.txt',
            'table_file' : f'{TABLES_PATH}\\table-step.pkl',
        }
    )

    tune_train = {
        'data_subset_name': DATA_SUBSET_GAIT_TRAINING,
        'normalization_settings': {
            'normalization_function': 'min_max_symmetrical',
            'normalization_mode': 'by_time_series'
        },
        'data_columns' : ['LAV', 'LAX', 'LAY', 'LAZ', 
                        'LRV', 'LRX', 'LRY', 'LRZ', 
                        'RAV', 'RAX', 'RAY', 'RAZ', 
                        'RRV', 'RRX', 'RRY', 'RRZ'],
        'label_columns' : ['LFA', 'RFA'],
        'batch_size' : 64,
        'shuffle' : True,
        'dtype' : {
                 'LAV': np.float64, 'LAX': np.float64, 'LAY': np.float64,
                 'LAZ': np.float64, 'LRV': np.float64, 'LRX': np.float64, 
                 'LRY': np.float64, 'LRZ': np.float64, 'RAV': np.float64, 
                 'RAX': np.float64, 'RAY': np.float64, 'RAZ': np.float64, 
                 'RRV': np.float64, 'RRX': np.float64, 'RRY': np.float64, 
                 'RRZ': np.float64, 'LFA': np.int8,    'RFA': np.int8
                 },
        'verbose' : True,
        'length' : 1024,
        'validation_generator' : False,
        'train_test_split' : 1.0,
        'sample_freq' : 512,
        'sample_offset' : 0,
        'caching' : True,
        'padding' : True,
        'pad_last_batch' : True
    }

    tune_val = {
        'data_subset_name': DATA_SUBSET_GAIT_VALIDATION,
        'normalization_settings': {
            'normalization_function': 'min_max_symmetrical',
            'normalization_mode': 'by_time_series'
        },
        'data_columns' : ['LAV', 'LAX', 'LAY', 'LAZ', 
                        'LRV', 'LRX', 'LRY', 'LRZ', 
                        'RAV', 'RAX', 'RAY', 'RAZ', 
                        'RRV', 'RRX', 'RRY', 'RRZ'],
        'label_columns' : ['LFA', 'RFA'],
        'batch_size' : 64,
        'shuffle' : False,
        'dtype' : {
                 'LAV': np.float64, 'LAX': np.float64, 'LAY': np.float64,
                 'LAZ': np.float64, 'LRV': np.float64, 'LRX': np.float64, 
                 'LRY': np.float64, 'LRZ': np.float64, 'RAV': np.float64, 
                 'RAX': np.float64, 'RAY': np.float64, 'RAZ': np.float64, 
                 'RRV': np.float64, 'RRX': np.float64, 'RRY': np.float64, 
                 'RRZ': np.float64, 'LFA': np.int8,    'RFA': np.int8
                 },
        'verbose' : True,
        'length' : 1024,
        'validation_generator' : True,
        'train_test_split' : 0.0,
        'sample_freq' : 512,
        'sample_offset' : 0,
        'caching' : True,
        'padding' : True,
        'pad_last_batch' : True
    }

    

    tune_train_left_foot = tune_train.copy()
    tune_train_left_foot.update({
        # data_columns is a list of all columns that are part of the X variable
        'data_columns' : ['LAV', 'LAX', 'LAY', 'LAZ', 
                        'LRV', 'LRX', 'LRY', 'LRZ'],
        # label_columns is a list of all columns that are part of the y variable
        'label_columns' : ['LFA'],
                # providing a data type for all columns prevents pandas from inferring the data type
        # itself, leading to possibly unexpected behavior
        'dtype' : {
                 'LAV': np.float64, 'LAX': np.float64, 'LAY': np.float64,
                 'LAZ': np.float64, 'LRV': np.float64, 'LRX': np.float64, 
                 'LRY': np.float64, 'LRZ': np.float64, 'LFA': np.int8,
                 },
    })

    tune_val_left_foot = tune_val.copy()
    tune_val_left_foot.update({
        # data_columns is a list of all columns that are part of the X variable
        'data_columns' : ['LAV', 'LAX', 'LAY', 'LAZ', 
                        'LRV', 'LRX', 'LRY', 'LRZ'],
        # label_columns is a list of all columns that are part of the y variable
        'label_columns' : ['LFA'],
                # providing a data type for all columns prevents pandas from inferring the data type
        # itself, leading to possibly unexpected behavior
        'dtype' : {
                 'LAV': np.float64, 'LAX': np.float64, 'LAY': np.float64,
                 'LAZ': np.float64, 'LRV': np.float64, 'LRX': np.float64, 
                 'LRY': np.float64, 'LRZ': np.float64, 'LFA': np.int8,
                 },
    })

    post_processing_train = tune_train.copy()
    post_processing_train.update({
        'normalization_settings': DataNormalizationParams.best4,
    })

    post_processing_val = tune_val.copy()
    post_processing_val.update({
        'normalization_settings': DataNormalizationParams.best4,
    })

    post_processing_train_decimal = tune_train.copy()
    post_processing_train_decimal.update({
        'normalization_settings': DataNormalizationParams.decimal_scaling,
    })

    post_processing_val_decimal = tune_val.copy()
    post_processing_val_decimal.update({
        'normalization_settings': DataNormalizationParams.decimal_scaling,
    })


    post_processing_train_min_max = tune_train.copy()
    post_processing_train_min_max.update({
        'normalization_settings': DataNormalizationParams.min_max,
    })

    post_processing_val_min_max = tune_val.copy()
    post_processing_val_min_max.update({
        'normalization_settings': DataNormalizationParams.min_max,
    })

    post_processing_train_meta = post_processing_train.copy()
    post_processing_train_meta.update({
        'meta': True,
        'normalization_settings': DataNormalizationParams.min_max,

    })

    post_processing_val_meta = post_processing_val.copy()
    post_processing_val_meta.update({
        'meta': True,
        'normalization_settings': DataNormalizationParams.min_max,
    })


    submission_val_data = tune_val.copy()
    submission_val_data.update(
        {
            'data_subset_name': DATA_SUBMISSION,
            'normalization_settings': DataNormalizationParams.best4,

        }
    )

    submission_train_data = tune_train.copy()
    submission_train_data.update(
        {
            'data_subset_name': DATA_GAIT,
            'normalization_settings': DataNormalizationParams.best4,
        }
    )
