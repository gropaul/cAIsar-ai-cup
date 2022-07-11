# Holds the parameter of a session including
# - augmentation config
# - training config
# - model config
# - data config
from data_generator.batch_generator_params import BatchGeneratorParams
from data_generator.data_augmenter_params import DataAugmenterParams
from model.ueber_net.ueber_net_params import UeberNetParams


class SessionParams():
    
    config_1 = {
        'session_param_id' : 1,
        'bg_train_config' : BatchGeneratorParams.post_processing_train,
        'bg_evaluation_config' : BatchGeneratorParams.post_processing_val,

        'data_augmenter_config' : DataAugmenterParams.best4,

        'model_config': UeberNetParams.best4,

        'training_config': {
            'epochs' : 37,
            'loss': 'dice_loss',
            'tversky_beta': 0.8,
            'learning_rate': 0.001,
            'optimizer': 'Adam',
            'lr_epoch_per_step': 5,
            'lr_increase_per_step': 0.5,
            'lr_number_of_steps': 3,
            'lr_start': 0.01
        }
    }

    config_1_left_foot = {
        'session_param_id' : 1,
        'bg_train_config' : BatchGeneratorParams.tune_train_left_foot,
        'bg_evaluation_config' : BatchGeneratorParams.tune_val_left_foot,

        'data_augmenter_config' : DataAugmenterParams.best4,

        'model_config': UeberNetParams.best4,

        'training_config': {
            'epochs' : 35,
            'loss': 'dice_loss',
            'tversky_beta': 0.8,
            'learning_rate': 0.001,
            'optimizer': 'Adam'
        }
    }

    config_2 = config_1.copy()
    config_2.update({
        'session_param_id' : 2,
        'model_config': UeberNetParams.best2,
    })

    config_21 = config_1.copy()
    config_21.update({
        'session_param_id' : 21,
        'model_config': UeberNetParams.best2_adaption,
    })

    config_22 = config_1.copy()
    config_22.update({
        'session_param_id' : 22,
        'model_config': UeberNetParams.default_adaption,
        'training_config': {
            'epochs' : 35,
            'loss': 'tversky_loss',
            'tversky_beta': 0.8,
            'learning_rate': 0.001,
            'optimizer': 'Adam'
        }
    })

    config_23 = config_1.copy()
    config_23.update({
        'session_param_id' : 23,
        'model_config': UeberNetParams.default,
        'training_config': {
            'epochs' : 35,
            'loss': 'tversky_loss',
            'tversky_beta': 0.8,
            'learning_rate': 0.001,
            'optimizer': 'Adam'
        }
    })

    # less training epochs
    config_231 = config_1.copy()
    config_231.update({
        'session_param_id' : 231,
        'model_config': UeberNetParams.default,
        'training_config': {
            'epochs' : 25,
            'loss': 'tversky_loss',
            'tversky_beta': 0.8,
            'learning_rate': 0.001,
            'optimizer': 'Adam'
        }
    })

    # no augmentation
    config_232 = config_23.copy()
    config_232.update({
        'session_param_id' : 232,
        'data_augmenter_config' : DataAugmenterParams.default,
    })

    config_3 = config_1.copy()
    config_3.update({
        'session_param_id' : 3,
        'model_config': UeberNetParams.best,
        'training_config': {
            'epochs' : 35,
            'loss': 'dice_loss',
            'tversky_beta': 0.8,
            'learning_rate': 0.001,
            'optimizer': 'Adam'
        }
    })

    
    config_3_left_foot = {
        'session_param_id' : 31,
        'bg_train_config' : BatchGeneratorParams.tune_train_left_foot,
        'bg_evaluation_config' : BatchGeneratorParams.tune_val_left_foot,

        'data_augmenter_config' : DataAugmenterParams.best4,

        'model_config': UeberNetParams.best4,

        'training_config': {
            'epochs' : 35,
            'loss': 'dice_loss',
            'tversky_beta': 0.8,
            'learning_rate': 0.001,
            'optimizer': 'Adam'
        }
    }

    config_5_tune = {
        'session_param_id' : 51,
        'bg_train_config' : BatchGeneratorParams.tune_train,
        'bg_evaluation_config' : BatchGeneratorParams.tune_val,

        'data_augmenter_config' : DataAugmenterParams.best4,

        'model_config': UeberNetParams.default,

        'training_config': {
            'epochs' : 35,
            'loss': 'dice_loss',
            'tversky_beta': 0.8,
            'learning_rate': 0.001,
            'optimizer': 'Adam'
        } 
    }

    config_4_metadata = {
       'session_param_id' : 4,
        'bg_train_config' : BatchGeneratorParams.post_processing_train_meta,
        'bg_evaluation_config' : BatchGeneratorParams.post_processing_val_meta,

        'data_augmenter_config' : DataAugmenterParams.best4,

        'model_config': UeberNetParams.default_with_meta,

        'training_config': {
            'epochs' : 35,
            'loss': 'dice_loss',
            'tversky_beta': 0.8,
            'learning_rate': 0.001,
            'optimizer': 'Adam',
            'lr_epoch_per_step': 5,
            'lr_increase_per_step': 0.5,
            'lr_number_of_steps': 3,
            'lr_start': 0.1
        } 
    }

    config_6 = {
        'session_param_id' : 6,
        'bg_train_config' : BatchGeneratorParams.post_processing_train,
        'bg_evaluation_config' : BatchGeneratorParams.post_processing_val,

        'data_augmenter_config' : DataAugmenterParams.default,

        'model_config': UeberNetParams.default,

        'training_config': {
            'epochs' : 36,
            'loss': 'tversky_loss',
            'tversky_beta': 0.8,
            'learning_rate': 0.001,
            'optimizer': 'Adam',
            'lr_epoch_per_step': 5,
            'lr_increase_per_step': 0.5,
            'lr_number_of_steps': 3,
            'lr_start': 0.1
        } 
    }

    # everything default but model config
    config_7 = {
        'session_param_id' : 7,
        'bg_train_config' : BatchGeneratorParams.post_processing_train,
        'bg_evaluation_config' : BatchGeneratorParams.post_processing_val,

        'data_augmenter_config' : DataAugmenterParams.default,

        'model_config': UeberNetParams.default,

        'training_config': {
            'epochs' : 36,
            'loss': 'tversky_loss',
            'tversky_beta': 0.8,
            'learning_rate': 0.001,
            'optimizer': 'Adam',
            'lr_epoch_per_step': 5,
            'lr_increase_per_step': 0.5,
            'lr_number_of_steps': 3,
            'lr_start': 0.1
        } 
    }

    # everything default but model config
    config_8 = {
        'session_param_id' : 8,
   
        'data_augmenter_config' : DataAugmenterParams.default,
        'bg_train_config' : BatchGeneratorParams.post_processing_train_decimal,
        'bg_evaluation_config' : BatchGeneratorParams.post_processing_val_decimal,

        'model_config': UeberNetParams.hyper_best,

        'training_config': {
            'epochs' : 30,
            'loss': 'tversky_loss',
            'tversky_beta': 0.8,
            'learning_rate': 0.001,
            'optimizer': 'Adam',
            'lr_epoch_per_step': 5,
            'lr_increase_per_step': 0.5,
            'lr_number_of_steps': 3,
            'lr_start': 0.1
        } 
    }

    # everything default but model config
    config_81 = config_8.copy()
    config_81.update({
        'session_param_id' : 81,
        'bg_train_config' : BatchGeneratorParams.post_processing_train_decimal,
        'bg_evaluation_config' : BatchGeneratorParams.post_processing_val_decimal,

        'data_augmenter_config' : DataAugmenterParams.default,
        'model_config': UeberNetParams.hyper_best_v2,
        'training_config': {
            'epochs' : 36,
            'loss': 'tversky_loss',
            'tversky_beta': 0.8,
            'learning_rate': 0.001,
            'optimizer': 'Adam',
            'lr_epoch_per_step': 5,
            'lr_increase_per_step': 0.5,
            'lr_number_of_steps': 3,
            'lr_start': 0.1
        } 
    })

    # best 100 run avg of run without Inceptions
    config_9 = config_8.copy()
    config_9.update({
        'session_param_id' : 9,
        'bg_train_config' : BatchGeneratorParams.post_processing_train_meta,
        'bg_evaluation_config' : BatchGeneratorParams.post_processing_val_meta,

        'data_augmenter_config' : DataAugmenterParams.default,
        'model_config': UeberNetParams.hyper_best_v3,
        'training_config': {
            'epochs' : 36,
            'loss': 'dice_loss',
            'tversky_beta': 0.8,
            'learning_rate': 0.001,
            'optimizer': 'Adam',
            'lr_epoch_per_step': 5,
            'lr_increase_per_step': 0.1,
            'lr_number_of_steps': 4,
            'lr_start': 0.01
        } 
    })

    # best 100 run avg of run without Inceptions, Version 2
    config_10 = config_8.copy()
    config_10.update({
        'session_param_id' : 1000,
        'bg_train_config' : BatchGeneratorParams.post_processing_train_min_max,
        'bg_evaluation_config' : BatchGeneratorParams.post_processing_val_min_max,

        'data_augmenter_config' : DataAugmenterParams.best4,
        'model_config': UeberNetParams.hyper_best_v4,
        'training_config': {
            'epochs' : 36,
            'loss': 'dice_loss',
            'tversky_beta': 0.8,
            'learning_rate': 0.001,
            'optimizer': 'RMSprop',
            'lr_epoch_per_step': 5,
            'lr_increase_per_step': 0.1,
            'lr_number_of_steps': 1,
            'lr_start': 0.01
        } 
    })

