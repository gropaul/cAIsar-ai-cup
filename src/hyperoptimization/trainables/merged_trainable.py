from typing import Any, Dict, List
import numpy as np
from ray import tune

from hyperoptimization.trainables.base_trainable import BaseTrainable

from model.metric_functions.cup_f1_score import cup_f1_score
from model.ueber_net.ueber_net_params import UeberNetParams

from data_generator.data_augmenter_params import DataAugmenterParams
from data_generator.data_augmenter import DataAugmenter
from data_generator.batch_generator_params import BatchGeneratorParams

import utils.util_functions as utils
from utils.util_functions import get_updated
from utils.errors import WrongDataTypeInSearchSpaceError

from session.session import Session
from session.cli_parser import CLIParser
from session.callbacks.learning_rate_scheduler import CustomLearningRateScheduler


class MergedTrainable(BaseTrainable):
    
    metric = 'f1score'
    mode = 'max'

    hyperopt_space = {

        ### Preprocessing Space ### 

        'normalization_settings': {
            'normalization_mode' : tune.choice(['by_time_series', 'by_all']),
            'normalization_function' : tune.choice([ 
                'min_max_symmetrical',
                'tanh_estimator',
                'standardization',
                'median_normalization',
                'sigmoid_normalization',
                'decimal_scaling_normalization',
            ]),
        },

        # Drift
        'drift_max': tune.uniform(0.0,0.3),
        'drift_points': tune.randint(4,100),
        'drift_kind': tune.choice(['additive', 'multiplicative']),
        'drift_prob': tune.uniform(0.0,1.0),

        # Noise
        'noise_scale': tune.uniform(0.0, 0.07),
        'noise_prob': tune.uniform(0.0,1.0),

        # Convolution
        'convolve_window_type': tune.choice([None,'bartlett','flattop','parzen','bohman']),
        'convolve_window_size': tune.randint(2,15),
        'convolve_prob': tune.uniform(0.0,1.0),

        # Dropout
        'dropout_percentage': tune.uniform(0.0,0.5),
        'dropout_prob': tune.uniform(0.0,1.0),
        'dropout_size': tune.randint(1,10),
        'dropout_fill': tune.choice(['ffill','bfill','mean',0.0]),

        # TimeWarp
        'time_warp_changes': tune.randint(1,50),
        'time_warp_max': tune.uniform(1.1,3),
        'time_warp_prob': tune.uniform(0.0,1.0),

        ### Training Space
        'lr_epoch_per_step' : tune.choice([5,9]),
        'lr_number_of_steps' : tune.choice([1,3,4]),
        'lr_start' : tune.choice([0.1, 0.01]),
        'lr_increase_per_step' : tune.choice([0.5, 0.1]),

        ### Uber Trainable Space ###

        'base_architecture' : tune.choice(['U-net', 'U-net++']),
        'additional_architectures' : {
            'attention' : tune.choice([False, True]),
            'dense' : tune.choice([False, True]),
            'inception' : tune.choice([False, True]),
            'residual' : tune.choice([False, True]),
        },
        'optimizer' : tune.choice(['Adadelta', 'Adagrad', 'Adam', 'Adamax', 'Ftrl', 'Nadam', 'RMSprop', 'SGD']),
        'loss' : tune.choice(['dice_loss', 'BinaryCrossentropy', 'tversky_loss']),
        'tversky_beta' : tune.uniform(0.0, 1.0),
        'base_filter_count' : tune.randint(4, 65),
        'backbone_length' : tune.randint(2, 7),
        'kernel_size' : tune.randint(2, 11),
        'dropout' : tune.uniform(0.0, 0.5),
        'n_fold_convolutions' : tune.randint(1, 5),
        'attention_kernel' : tune.randint(1, 7),
        'attention_intermediate' : tune.uniform(0.0, 1.0),
        'inception_kernel_size' : tune.randint(3, 7),
        'meta' : tune.choice([False, True]),
        'meta_dropout' : tune.uniform(0.0, 1.0)
    }
    
    def parse_config(self, config: Dict[str, Any]):
        '''
        parses the config: adapted to deal with all 
        available spaces and their specifics
        NOTE: for understandability the order of parsed args
              should match the order of definition in the search space
        NOTE: for readability parse/check for types in 
              the order str > float > np.float64 (Bayes) > bool > int
        '''
        

        ### Ueber Net Config Parser ### 
        base_architecture = config['base_architecture']
        if type(base_architecture) == str:
            pass
        elif (type(base_architecture) == float) or (type(base_architecture) == np.float64):
            base_architectures = {
                0 : 'U-net',
                1 : 'U-net++'
            }
            config['base_architecture'] = base_architectures[int(base_architecture)]
        else:
            raise WrongDataTypeInSearchSpaceError(
                variable='base_architecture',
                value=base_architecture,
                data_type=type(base_architecture),
                expected_types=[str, float, np.float64]
            )
        
        additional_architectures: List[str] = []
        for add_arch in config['additional_architectures'].keys():
            if (type(config['additional_architectures'][add_arch]) == float) or (type(config['additional_architectures'][add_arch]) == np.float64):
                truth_values = {
                    0: False,
                    1: True
                }
                if truth_values[int(config['additional_architectures'][add_arch])]:
                    additional_architectures.append(add_arch.capitalize())
            elif type(config['additional_architectures'][add_arch]) == bool:
                if config['additional_architectures'][add_arch]:
                    additional_architectures.append(add_arch.capitalize())
            else:
                raise WrongDataTypeInSearchSpaceError(
                    variable=f'additional_architectures/{add_arch}',
                    value=add_arch,
                    data_type=type(config['additional_architectures'][add_arch]),
                    expected_types=[float, np.float64, bool]
                )
        config['additional_architectures'] = additional_architectures

        optimizer = config['optimizer']
        if type(optimizer) == str:
            pass
        elif (type(optimizer) == float) or (type(optimizer) == np.float64):
            optimizers = {
                0: 'Adadelta',
                1: 'Adagrad',
                2: 'Adam',
                3: 'Adamax',
                4: 'Ftrl',
                5: 'Nadam',
                6: 'RMSprop',
                7: 'SGD',
            }
            config['optimizer'] = optimizers[int(optimizer)]
        else:
            raise WrongDataTypeInSearchSpaceError(
                variable='optimizer',
                value=optimizer,
                data_type=type(optimizer),
                expected_types=[str, float, np.float64]
            )

    

        loss = config['loss']
        if type(loss) == str:
            pass
        elif (type(loss) == float) or (type(loss) == np.float64):
            losses = {
                0: 'dice_loss',
                1: 'BinaryCrossentropy',
                2: 'tversky_loss'
            }
            config['loss'] = losses[int(loss)]
        else:
            raise WrongDataTypeInSearchSpaceError(
                variable='loss',
                value=loss,
                data_type=type(loss),
                expected_types=[str, float, np.float64]
            )

        base_filter_count = config['base_filter_count']
        if (type(base_filter_count) == float) or (type(base_filter_count) == np.float64):
            config['base_filter_count'] = int(base_filter_count)
        elif type(base_filter_count) == int:
            pass
        else:            
            raise WrongDataTypeInSearchSpaceError(
                variable='base_filter_count',
                value=base_filter_count,
                data_type=type(base_filter_count),
                expected_types=[float, np.float64, int]
            )

        backbone_length = config['backbone_length']
        if (type(backbone_length) == float) or (type(backbone_length) == np.float64):
            config['backbone_length'] = int(backbone_length)
        elif type(backbone_length) == int:
            pass
        else:
            raise WrongDataTypeInSearchSpaceError(
                variable='backbone_length',
                value=backbone_length,
                data_type=type(backbone_length),
                expected_types=[float, np.float64, int]
            )

        kernel_size = config['kernel_size']
        if (type(kernel_size) == float) or (type(kernel_size) == np.float64):
            kernel_size = int(config['kernel_size'])
        elif type(kernel_size) == int:
            pass
        else:
            raise WrongDataTypeInSearchSpaceError(
                variable='kernel_size',
                value=kernel_size,
                data_type=type(kernel_size),
                expected_types=[float, np.float64, int]
            )
        config['kernel_size'] = {-1: kernel_size}

        config['dropout'] = {-1: config['dropout']}
        
        n_fold_convolutions = config['n_fold_convolutions']
        if (type(n_fold_convolutions) == float) or (type(n_fold_convolutions) == np.float64):
            config['n_fold_convolutions'] = int(n_fold_convolutions)
        elif type(n_fold_convolutions) == int:
            pass
        else:
            raise WrongDataTypeInSearchSpaceError(
                variable='n_fold_convolutions',
                value=n_fold_convolutions,
                data_type=type(n_fold_convolutions),
                expected_types=[np.float64, int]
            )

        attention_kernel = config['attention_kernel']
        if (type(attention_kernel) == float) or (type(attention_kernel) == np.float64):
            attention_kernel = int(attention_kernel)
        elif type(attention_kernel) == int:
            pass
        else:
            raise WrongDataTypeInSearchSpaceError(
                variable='attention_kernel',
                value=attention_kernel,
                data_type=type(attention_kernel),
                expected_types=[float, np.float64, int]
            )
        config['attention_kernel'] = {-1: attention_kernel}

        config['attention_intermediate'] = {-1: config['attention_intermediate']}

        inception_kernel_size = config['inception_kernel_size']
        if (type(inception_kernel_size) == float) or (type(inception_kernel_size) == np.float64):
            inception_kernel_size = int(inception_kernel_size)
        elif type(inception_kernel_size) == int:
            pass
        else:
            raise WrongDataTypeInSearchSpaceError(
                variable='inception_kernel_size',
                value=inception_kernel_size,
                data_type=type(inception_kernel_size),
                expected_types=[float, np.float64, int]
            )
        config['inception_kernel_size'] = {-1: inception_kernel_size}

        ### Preprocessing Config Parser ### 

        # no parsing needed for now

        return config


    def setup(self, config) -> None:
        config = self.parse_config(config)

        # CLI Config

        cli_args = ['-id', '100000000000', 
            '-l', config['loss'], '-lp', 'beta', str(config['tversky_beta']), 
            '-lr', str(0.001), '-e', '10', '-o', str(config['optimizer']), 
            '-mc', 'UeberNet', '-mp', 'default',
            '-tg', 'tune_train', '-vg', 'tune_val',
            '-svtb', 'False', '-svcp', 'False',
            '-svhis', 'False', '-svm', 'False',
        ]

        parser = CLIParser()
        cli_args = parser.parse_args(args=cli_args)


        ### Ueber Net Setup ### 

        model_dict = utils.get_partial_dict(dictionary=config, keys=[
            'base_architecture', 'additional_architectures',
            'base_filter_count', 'backbone_length', 'kernel_size',
            'dropout', 'n_fold_convolutions', 'attention_kernel', 
            'attention_intermediate', 'inception_kernel_size', 'meta_dropout'])

        if config['meta']:
            model_params = get_updated(UeberNetParams.default, **model_dict, meta_length=14)
        else:
            model_params = get_updated(UeberNetParams.default, **model_dict)

        ### Preprocessing Setup ### 

        # Augmentation Setup
        augmentation_dict = utils.get_partial_dict(dictionary=config, keys=[
            'noise_scale',
            'noise_prob',
            'convolve_window_type',
            'convolve_window_size',
            'convolve_prob',
            'dropout_prob',
            'dropout_size',
            'dropout_fill',
            'drift_max',
            'drift_points',
            'drift_kind',
            'drift_prob',
            'time_warp_changes',
            'time_warp_max',
            'time_warp_prob',
        ])

        aug_params = get_updated(DataAugmenterParams.default, **augmentation_dict)
        augmenter = DataAugmenter.get_custom(**aug_params)
                
        # Batch Generator Setup 
        tune_train = BatchGeneratorParams.tune_train.copy()
        tune_train.update({
            'normalization_settings' : config['normalization_settings'] 
        })

        tune_val = BatchGeneratorParams.tune_val.copy()
        tune_val.update({
            'normalization_settings' : config['normalization_settings'] 
        })

        ### Session setup ###

        # create session
        self.session = Session(args=cli_args, tune=True, auto_init=False)

        meta: bool = config['meta']
        # apply batch generator config 
        self.session.training_generator_params = get_updated(params=tune_train, augmenter=augmenter, meta=meta)
        self.session.validation_generator_params = get_updated(params=tune_val, validation_generator=True, meta=meta)

        # apply model and augmenter
        self.session.model_params = model_params
        self.session.augmenter = augmenter

        self.session.initialize()

        # add dynamic learning rate
        lr_per_epoch = self.get_learning_rate(
            config['lr_epoch_per_step'], 
            config['lr_number_of_steps'], 
            config['lr_increase_per_step'],
            config['lr_start'] 
        )
        self.session.callbacks.append(
            CustomLearningRateScheduler(lr_per_epoch=lr_per_epoch),
        )


    def get_learning_rate(self,epoch_per_step, number_of_steps, increase_per_step,start_lr ):
        return [(i * epoch_per_step, round(start_lr * pow(increase_per_step,i), ndigits=6)) for i in range(number_of_steps)]

    def step(self):
        self.session.step(step_size=2)
        y_pred, y_true = self.session.get_predictions_as_ts()
        fscore, precision, recall = cup_f1_score(y_pred=y_pred, y_true=y_true)
        return {MergedTrainable.metric : fscore, 'precision' : precision, 'recall' : recall}
    
    def save_checkpoint(self, checkpoint_dir):
        path = self.session.save_keras_model(model_path=checkpoint_dir)
        return path
    
    def load_checkpoint(self, path):
        self.session.load_keras_model(model_path=path)
    
    # no reset function, as batch generator must often be swapped due to different normalization settings

    
