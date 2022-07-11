
from typing import Dict
from session_validation.session_params import SessionParams
from model.ueber_net.ueber_net_params import UeberNetParams
from data_generator.data_augmenter_params import DataAugmenterParams
from data_generator.batch_generator_params import BatchGeneratorParams
from utils.SYSCONFIG import DATA_SUBSET_GAIT_TRAINING
from utils.SYSCONFIG import DATA_SUBSET_GAIT_VALIDATION

def get_session_config_from_hyper_params(
    hyper_params: Dict, 
    epochs: int, 
    data_subset_train: str = DATA_SUBSET_GAIT_TRAINING,
    data_subset_val: str = DATA_SUBSET_GAIT_VALIDATION,
):
    cfg = SessionParams.config_1.copy()

    cfg['model_config'] = get_model_config_from_hyper_params(hyper_params)
    cfg['data_augmenter_config'] = get_aug_config_from_hyper_params(hyper_params)
    cfg['training_config'] = get_train_config_from_hyper_params(hyper_params, epochs)
    cfg['bg_evaluation_config'] = get_bg_val_config_from_hyper_params(hyper_params, data_subset_val)
    cfg['bg_train_config'] = get_bg_train_config_from_hyper_params(hyper_params, data_subset_train)
    return cfg

def get_bg_train_config_from_hyper_params(hyper_params, data_subset_name):
    bg_config = BatchGeneratorParams.tune_train
 
    bg_config['meta'] = hyper_params['meta']
    bg_config['normalization_settings'] =  {
        'normalization_function': hyper_params['normalization_settings_normalization_function'],
        'normalization_mode': hyper_params['normalization_settings_normalization_mode']
    }

    bg_config['data_subset_name'] = data_subset_name
    return bg_config


def get_bg_val_config_from_hyper_params(hyper_params, data_subset_name):
    bg_config = BatchGeneratorParams.tune_val
 
    bg_config['meta'] = hyper_params['meta']
    bg_config['normalization_settings'] =  {
        'normalization_function': hyper_params['normalization_settings_normalization_function'],
        'normalization_mode': hyper_params['normalization_settings_normalization_mode']
    }

    bg_config['data_subset_name'] = data_subset_name
    return bg_config


def get_train_config_from_hyper_params(hyper_params, epochs):
    training_config = {}
    training_config['epochs'] = epochs
    training_config['loss'] = hyper_params['loss']
    training_config['tversky_beta'] =  hyper_params['tversky_beta']
    training_config['learning_rate'] =  0.01
    training_config['optimizer'] =  hyper_params['optimizer']

    training_config['lr_epoch_per_step'] =  hyper_params['lr_epoch_per_step']
    training_config['lr_increase_per_step'] =  hyper_params['lr_increase_per_step']
    training_config['lr_number_of_steps'] =  hyper_params['lr_number_of_steps']
    training_config['lr_start'] =  hyper_params['lr_start']

    return training_config


def get_model_config_from_hyper_params(hyper_params):
    model_cfg = UeberNetParams.default.copy()
    model_cfg['base_architecture'] = hyper_params['base_architecture']
    
    model_cfg['additional_architectures'] = []
    if hyper_params['additional_architectures_attention']: 
        model_cfg['additional_architectures'].append('Attention')
    if hyper_params['additional_architectures_dense']: 
        model_cfg['additional_architectures'].append('Dense')
    if hyper_params['additional_architectures_inception']: 
        model_cfg['additional_architectures'].append('Inception')
    if hyper_params['additional_architectures_residual']: 
        model_cfg['additional_architectures'].append('Residual')

    model_cfg['base_filter_count'] = hyper_params['base_filter_count']
    model_cfg['backbone_length'] = hyper_params['backbone_length']

    model_cfg['kernel_size'] = {-1: hyper_params['kernel_size']}
    model_cfg['dropout'] = {-1: hyper_params['dropout']}

    model_cfg['n_fold_convolutions'] = hyper_params['n_fold_convolutions']

    model_cfg['attention_kernel'] = {-1: hyper_params['attention_kernel']}
    model_cfg['attention_intermediate'] = {-1: hyper_params['attention_intermediate']}
    
    model_cfg['inception_kernel_size'] = {-1: hyper_params['inception_kernel_size']}
    
    model_cfg['meta_length'] = hyper_params['inception_kernel_size']
    model_cfg['meta_dropout'] = hyper_params['meta_dropout']
    model_cfg['post_dense_meta_dropout'] = {-1: 0.0}

    if hyper_params['meta']:
        model_cfg['meta_length'] = 14
    else:
        model_cfg['meta_length'] = 0

    return model_cfg


def get_aug_config_from_hyper_params(hyper_params):
    auf_cfg = DataAugmenterParams.best

    auf_cfg['drift_max'] = hyper_params['drift_max']
    auf_cfg['drift_points'] = hyper_params['drift_points']
    auf_cfg['drift_kind'] = hyper_params['drift_kind']
    auf_cfg['drift_prob'] = hyper_params['drift_prob']

    auf_cfg['noise_scale'] = hyper_params['noise_scale']
    auf_cfg['noise_prob'] = hyper_params['noise_prob']

    auf_cfg['convolve_window_type'] = hyper_params['convolve_window_type']
    auf_cfg['convolve_window_size'] = hyper_params['convolve_window_size']
    auf_cfg['convolve_prob'] = hyper_params['convolve_prob']

    auf_cfg['dropout_percentage'] = hyper_params['dropout_percentage']
    auf_cfg['dropout_prob'] = hyper_params['dropout_prob']
    auf_cfg['dropout_size'] = hyper_params['dropout_size']
    auf_cfg['dropout_fill'] = hyper_params['dropout_fill']

    auf_cfg['time_warp_changes'] = hyper_params['time_warp_changes']
    auf_cfg['time_warp_max'] = hyper_params['time_warp_max']
    auf_cfg['time_warp_prob'] = hyper_params['time_warp_prob']

    return auf_cfg