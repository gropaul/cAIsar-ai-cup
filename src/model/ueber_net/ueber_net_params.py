from typing import Any, Dict
import utils.util_functions as utils

class UeberNetParams:
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
        'base_architecture' : 'U-net',
        'additional_architectures' : [],
        
        # NOTE: params inherent to the architecture and the problem
        'base_filter_count' : 16, 
        'backbone_length' : 5, 
        'concat_axis' : 2,

        # number of anomaly classes encoded as CLASS_COUNT; might exceed 1
        # as e. g. seasonal, time-independent etc. anomalies are differentiated
        'class_count' : 2, 
        
        # NOTE: params of the respective nodes in the architecture,
        # see paper and docs for precise meaning 
        'kernel_size' : {-1 : 3}, 
        'pool_size' : {-1 : 4},
        'dropout' : {-1 : 0.0},
        'strides' : {-1 : 1},

        # n-fold convolutions
        'n_fold_convolutions' : 2,

        # Attention params
        'attention_kernel' : {-1 : 3},
        'attention_intermediate' : {-1 : 0.2},

        # Inception params
        'inception_kernel_size' : {-1 : 3},

        # meta params
        'meta_length' : 0,
        'meta_dropout' : 0.0,
        'post_dense_meta_dropout' : {-1 : 0.0}
    }

    default_with_meta = default.copy()
    default_with_meta.update({
        'meta_length': 14,
    })


    default_adaption = default.copy()
    default_adaption.update({
        'additional_architectures' : ['Dense','Inception','Residual']
    })

    # best config (21.05.2022)
    best = default.copy()
    best.update(
        {
        'base_architecture' : 'U-net++',

        'additional_architectures' : ['Dense','Inception','Residual'],
        
        'base_filter_count' : 5, 
        'backbone_length' : 4, 
        'concat_axis' : 2,

        'class_count' : 2, 
        
        'kernel_size' : {-1 : 9}, 
        'pool_size' : {-1 : 4},
        'dropout' : {-1 : 0.125},
        'strides' : {-1 : 1},

        'n_fold_convolutions' : 1,

        'attention_kernel' : {-1 : 3},
        'attention_intermediate' : {-1 : 0.2},

        'inception_kernel_size' : {-1 : 3},
        
        'meta_length' : 0,
        'meta_dropout' : 0.0
    })

    # best config (15.06.2022)
    best2 = default.copy()
    best2.update(
        {
        'base_architecture' : 'U-net++',

        'additional_architectures' : ['Attention','Dense','Inception','Residual'],
        
        'base_filter_count' : 50, 
        'backbone_length' : 4, 
        'concat_axis' : 2,

        'class_count' : 2, 
        
        'kernel_size' : {-1 : 10}, 
        'pool_size' : {-1 : 4},
        'dropout' : {-1 : 0.2},
        'strides' : {-1 : 1},

        'n_fold_convolutions' : 1,

        # 'multi_heads' : {-1 : 1},
        # 'attention_key_dim' : {-1:8},
        'attention_kernel' : {-1 : 2},
        'attention_intermediate' : {-1 : 0.9325},


        'inception_kernel_size' : {-1 : 6},
    })

      # best config (15.06.2022)
    best2_adaption = best2.copy()
    best2_adaption.update(
        {

        'base_filter_count' : 5,         
        'kernel_size' : {-1 : 5}, 
        'pool_size' : {-1 : 4},
        'dropout' : {-1 : 0.2},
        'strides' : {-1 : 1},

        'n_fold_convolutions' : 1,

        'attention_kernel' : {-1 : 2},
        'attention_intermediate' : {-1 : 0.3},

        'inception_kernel_size' : {-1 : 3},
    })

    # best config by 22.06.2022 - Created by using the config of the best run 
    # of the hyperoptimization
    best3 = default.copy()
    best3.update(
        {
        'base_architecture' : 'U-net',

        'additional_architectures' : ['Dense'],
        
        'base_filter_count' : 45, 
        'backbone_length' : 4, 
        'concat_axis' : 2,

        'class_count' : 2, 
        
        'kernel_size' : {-1 : 4}, 
        'pool_size' : {-1 : 4},
        'dropout' : {-1 : 0.17},
        'strides' : {-1 : 1},

        'n_fold_convolutions' : 4,

        'attention_kernel' : {-1 : 6},
        'attention_intermediate' : {-1 : 0.11},

        'inception_kernel_size' : {-1 : 5},
    })

    # best config by 24.06.2022 - Created by looking at the avg config of the top 30 runs
    best4 = default.copy()
    best4.update(
        {
        'base_architecture' : 'U-net++',

        'additional_architectures' : ['Attention', 'Dense'],
        
        'base_filter_count' : 27, 
        'backbone_length' : 2, 
        'concat_axis' : 2,

        'class_count' : 2, 
        
        'kernel_size' : {-1 : 10}, 
        'pool_size' : {-1 : 4},
        'dropout' : {-1 : 0.17},
        'strides' : {-1 : 1},

        'n_fold_convolutions' : 4,

        'attention_kernel' : {-1 : 2},
        'attention_intermediate' : {-1 : 0.11},

        'inception_kernel_size' : {-1 : 4},
    })

    hyper_best = default.copy()
    hyper_best.update({
        'base_architecture' : 'U-net',

        'additional_architectures' : ['Attention', 'Dense', 'Attention' ],
        
        'base_filter_count' : 16, 
        'backbone_length' : 4, 
        'concat_axis' : 2,

        'class_count' : 2, 
        
        'kernel_size' : {-1 : 10}, 
        'pool_size' : {-1 : 4},
        'dropout' : {-1 : 0.17},
        'strides' : {-1 : 1},

        'n_fold_convolutions' : 4,

        'attention_kernel' : {-1 : 2},
        'attention_intermediate' : {-1 : 0.5},

        'inception_kernel_size' : {-1 : 6},
    })

    hyper_best_v2 = default.copy()
    hyper_best_v2.update({
        'base_architecture' : 'U-net',

        'additional_architectures' : ['Dense', 'Attention' ],
        
        'base_filter_count' : 50, 
        'backbone_length' : 4, 
        'concat_axis' : 2,

        'class_count' : 2, 
        
        'kernel_size' : {-1 : 8}, 
        'pool_size' : {-1 : 4},
        'dropout' : {-1 : 0.3},
        'strides' : {-1 : 1},

        'n_fold_convolutions' : 1,

        'attention_kernel' : {-1 : 4},
        'attention_intermediate' : {-1 : 0.5},

        'inception_kernel_size' : {-1 : 6},
    })

    # best 100 run avg of run without Inceptions
    hyper_best_v3 = default.copy()
    hyper_best_v3.update({
        'base_architecture' : 'U-net',

        'additional_architectures' : ['Dense', 'Attention' ],
        
        'base_filter_count' : 40, 
        'backbone_length' : 6, 
        'concat_axis' : 2,

        'class_count' : 2, 
        
        'kernel_size' : {-1 : 8}, 
        'pool_size' : {-1 : 4},
        'dropout' : {-1 : 0.25},
        'strides' : {-1 : 1},

        'n_fold_convolutions' : 1,

        'attention_kernel' : {-1 : 4},
        'attention_intermediate' : {-1 : 0.8},

        'inception_kernel_size' : {-1 : 6},

        # meta params
        'meta_length' : 14,
        'meta_dropout' : 0.4,
    })

    
    # best 100 run avg of run without Inceptions, Vesrion 2
    hyper_best_v4 = default.copy()
    hyper_best_v4.update({
        'base_architecture' : 'U-net',

        'additional_architectures' : [],
        
        'base_filter_count' : 37, 
        'backbone_length' : 6, 
        'concat_axis' : 2,

        'class_count' : 2, 
        
        'kernel_size' : {-1 : 7}, 
        'pool_size' : {-1 : 4},
        'dropout' : {-1 : 0.3},
        'strides' : {-1 : 1},

        'n_fold_convolutions' : 1,

        'attention_kernel' : {-1 : 3},
        'attention_intermediate' : {-1 : 0.6},

        'inception_kernel_size' : {-1 : 5},

        # meta params
        'meta_length' : 0,
        'meta_dropout' : 0.0,
    })



    