class UnetPlusPlusParams:
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
        'backbone_length' : 5, 

        # number of anomaly classes encoded as CLASS_COUNT; might exceed 1
        # as e. g. seasonal, time-independent etc. anomalies are differentiated
        'class_count' : 2, 
        
        # NOTE: params of the respective nodes in the architecture,
        # see paper and docs for precise meaning 
        'kernel_size' : {-1 : 3}, 
        'pool_size' : {-1 : 4},
        'dropout' : {-1 : 0.0},
        'strides' : {-1 : 1},

        # double convolutions
        'double_convolutions' : False
    }