# used to execute a session using a configuration of session_params.py
from session.callbacks.learning_rate_scheduler import CustomLearningRateScheduler
from session_validation.callbacks.validation_callback import ValidationCallback, get_validation_callback_data_path
from session_validation.session_params import SessionParams
from utils.util_functions import get_updated
from utils.SYSCONFIG import DATA_GAIT, DATA_SUBMISSION, DATA_SUBSET_GAIT_TRAINING

import pandas as pd

def execute_training(
    session_param_id : int,
    bg_train_config, bg_evaluation_config, 
    data_augmenter_config, 
    model_config,
    training_config,
):

    # 4) Model Training Config
    EPOCHS = training_config['epochs']

    # Train model

    from data_generator.data_augmenter import DataAugmenter
    from session.session import Session
    from session.cli_parser import CLIParser
    
    # date is not a good id as it makes training unresumable
    id = f"44444444{session_param_id:04d}"

    cli_args = [
        '-id', id, ## 7 is already trained with .best 
        '-l', str(training_config['loss']), '-lp', 'beta', str( training_config['tversky_beta']), 
        '-lr', str(training_config['learning_rate']), '-e', str(EPOCHS), '-o', training_config['optimizer'],  # SGD hat extrem lange gebraucht, Adam ist super
        '-utts', 'False',
        '-mc', 'UeberNet', '-mp', 'default',
        '-tg', 'submission_train_data', '-vg', 'submission_val_data',
        '-svtb', 'True', '-svcp', 'True',
        '-svhis', 'True', '-svm', 'True',
    ]

    parser = CLIParser()
    cli_args = parser.parse_args(args=cli_args)

    aug_params = data_augmenter_config
    augmenter = DataAugmenter.get_custom(**aug_params)

    session = Session(args=cli_args, tune=False, auto_init=False)
    session.model_params = model_config

    # change batch generator config
    bg_train_config['data_subset_name'] = DATA_GAIT
    bg_evaluation_config['data_subset_name'] = DATA_SUBMISSION


    # apply batch generator config 
    session.training_generator_params = get_updated(params=bg_train_config, augmenter=augmenter)
    session.validation_generator_params = get_updated(params=bg_evaluation_config, validation_generator=True)

    # apply augmentation config
    session.augmenter = augmenter

    session.initialize()

       # add dynamic learning rate
    lr_per_epoch = get_learning_rate(
        training_config['lr_epoch_per_step'], 
        training_config['lr_number_of_steps'], 
        training_config['lr_increase_per_step'],
        training_config['lr_start'] 
    )

    session.callbacks.append(
        CustomLearningRateScheduler(lr_per_epoch=lr_per_epoch)
    )
    session.callbacks.append(
        ValidationCallback(session=session, use_training=True)
    )

    validation_data_path = get_validation_callback_data_path(session.id)

    session.execute()

    return session, pd.read_csv(validation_data_path)

def get_learning_rate(epoch_per_step, number_of_steps, increase_per_step,start_lr ):
    return [(i * epoch_per_step, round(start_lr * pow(increase_per_step,i), ndigits=7)) for i in range(number_of_steps)]


if __name__ == '__main__':
    execute_training(**SessionParams.config_1)
    