from typing import Any, Dict
from data_generator.batch_generator_params import BatchGeneratorParams
from ray import tune
from hyperoptimization.trainables.base_trainable import BaseTrainable
from session.session import Session
from session.cli_parser import CLIParser

from model.metric_functions.cup_f1_score import cup_f1_score
from model.ueber_net.ueber_net_params import UeberNetParams

from data_generator.data_augmenter_params import DataAugmenterParams
from data_generator.data_augmenter import DataAugmenter
from soupsieve import select

import utils.util_functions as utils
from utils.util_functions import get_updated

class PreprocessingTrainable(BaseTrainable):


    def __init__(self, config: Dict[str, Any] = None, logger_creator = None, remote_checkpoint_dir = None, sync_function_tpl = None):
        print("PreProcessingTrainable INIT")

        super().__init__(config, logger_creator, remote_checkpoint_dir, sync_function_tpl)
    
    metric = 'f1score'
    mode = 'max'

    hyperopt_space = {

            'seed': tune.choice([42,None]),
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
            

        return config

    def get_current_session(self):
        return self.session

    def setup(self, config) -> None:
        config = self.parse_config(config)

        # setting batch generator config

        cli_args = ['-id', '100000000000', 
            '-l', 'tversky_loss', '-lp', 'beta', str(0.8), 
            '-lr', str(0.001), '-e', '10', '-o', 'Adam', 
            '-mc', 'UeberNet', '-mp', 'default',
            '-tg', 'tune_train', '-vg', 'tune_val',
            '-svtb', 'False', '-svcp', 'False',
            '-svhis', 'False', '-svm', 'False',
        ]

        #print(cli_args)

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
            'seed',
        ])

        parser = CLIParser()
        cli_args = parser.parse_args(args=cli_args)

        aug_params = get_updated(DataAugmenterParams.default, **augmentation_dict)
        augmenter = DataAugmenter.get_custom(**aug_params)

        
        # configure BatchGeneratorParams
        tune_train = BatchGeneratorParams.tune_train.copy()
        tune_train.update({
            'normalization_settings' : config['normalization_settings'] 
        })

        tune_val = BatchGeneratorParams.tune_val.copy()
        tune_val.update({
            'normalization_settings' : config['normalization_settings'] 
        })

        # create session
        self.session = Session(args=cli_args, tune=True, auto_init=False)

        # apply batch generator config 
        self.session.training_generator_params = get_updated(params=tune_train, augmenter=augmenter)
        self.session.validation_generator_params = get_updated(params=tune_val, validation_generator=True)

        # apply model and augmenter
        self.session.model_params = UeberNetParams.best
        self.session.augmenter = augmenter
        self.session.initialize()

        print("PreProcessingTrainable Setup")

    def step(self):
        self.session.step(step_size=2)
        y_pred, y_true = self.session.get_predictions_as_ts()
        fscore, precision, recall = cup_f1_score(y_pred=y_pred, y_true=y_true)
        return {PreprocessingTrainable.metric : fscore, 'precision' : precision, 'recall' : recall}
    
    def save_checkpoint(self, checkpoint_dir):
        path = self.session.save_keras_model(model_path=checkpoint_dir)
        return path
    
    def load_checkpoint(self, path):
        self.session.load_keras_model(model_path=path)
   
    # no reset function, as batch generator must often be swapped due to different normalization settings

    
