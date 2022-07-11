import argparse

from utils.SYSCONFIG import PROCESSOR_CORES, LOGS_PATH, MODELS_PATH, ID_MARKER, PLATFORM
from utils.SYSCONFIG import PROCESSOR_CORES, LOGS_PATH, MODELS_PATH, ID_MARKER, PLATFORM
from session.store_single_value_action import StoreSingleValueAction
from session.session import Session

class CLIParser(argparse.ArgumentParser):

    def __init__(self) -> argparse.ArgumentParser:
        description: str = 'CLI Interface to start a single training session.'
        super().__init__(description=description)

        # training parameter group
        tr_group_req = self.add_argument_group('required TRAINING arguments')
        tr_group_req.add_argument('-id', '--id', help='ID of the run as 12-digit integer. MUST be provided.', type=int, nargs=1, required=True, action=StoreSingleValueAction)
        tr_group_req.add_argument('-l', '--loss', help='Loss function to train the model with. MUST be specified.', type=str, nargs=1, required=True, action=StoreSingleValueAction, choices=Session.losses.keys())
        
        tr_group_opt = self.add_argument_group('optional TRAINING arguments')
        tr_group_opt.add_argument('-lr', '--learning_rate', help='Learning rate of the model. If any, at least one must be specified. Number of arguments MUST match EPOCHS.', type=float, nargs=1, default=0.001, action=StoreSingleValueAction)
        tr_group_opt.add_argument('-e', '--epochs', help='Number of epochs to train the model for. If any, at least one must be specified. Number of arguments MUST match LEARNING_RATE.', type=int, nargs=1, default=30, action=StoreSingleValueAction)
        tr_group_opt.add_argument('-o', '--optimizer', help='Optimizer to use during training.', type=str, nargs=1, default='Adam', action=StoreSingleValueAction, choices=Session.optimizers.keys())
        tr_group_req.add_argument('-lp', '--loss_params', help='Loss function params passed to the loss function.', type=str, nargs='*', default=[])
        tr_group_opt.add_argument('-m', '--metrics', help='Metrics to track during training.', type=str, nargs=1, default='default', action=StoreSingleValueAction, choices=Session.metrics.keys())
        tr_group_opt.add_argument('-sh', '--shuffle', help='Shuffle the data set/batch generator param.', type=str, nargs=1, default='True', action=StoreSingleValueAction, choices=Session.shuffle.keys())


        # sysconfig group
        sysconfig_group_req = self.add_argument_group('required SYSCONFIG arguments')
        
        sysconfig_group_opt = self.add_argument_group('optional SYSCONFIG arguments')
        sysconfig_group_opt.add_argument('-pc', '--processor_cores', help='Number of cores to use for multiprocessing.', type=int, nargs=1, default=PROCESSOR_CORES, action=StoreSingleValueAction)
        sysconfig_group_opt.add_argument('-rt', '--retries', help='Number of retries the session attempts upon failure.', type=int, nargs=1, default=3, action=StoreSingleValueAction)
        csvf_default = f'{LOGS_PATH}\\csv-{ID_MARKER}.log' if PLATFORM == 'WINDOWS' else f'{LOGS_PATH}/csv-{ID_MARKER}.log'
        sysconfig_group_opt.add_argument('-csvf', '--csv_callback_file', help='File to store the results of the csv log callback.', type=str, nargs=1, default=csvf_default, action=StoreSingleValueAction)
        logf_default = f'{LOGS_PATH}\\logging_{ID_MARKER}.out' if PLATFORM == 'WINDOWS' else f'{LOGS_PATH}/logging_{ID_MARKER}.out'
        sysconfig_group_opt.add_argument('-logf', '--log_file', help='File to save the training log to.', type=str, nargs=1, default=logf_default, action=StoreSingleValueAction)
        sysconfig_group_opt.add_argument('-clf', '--config_log_file', help='File to store the configuration of the run as json', type=str, nargs=1, default=f'{LOGS_PATH}\\conf-{ID_MARKER}.json', action=StoreSingleValueAction)

        sysconfig_group_opt.add_argument('-svtb', '--save_tensorboard_callback', help='Whether to store the output of the tensorboard callback.', type=str, nargs=1, default='True', action=StoreSingleValueAction, choices=Session.save_options)
        tbd_default = f'{LOGS_PATH}\\tb-{ID_MARKER}\\' if PLATFORM == 'WINDOWS' else f'{LOGS_PATH}/tb-{ID_MARKER}/'
        sysconfig_group_opt.add_argument('-tbd', '--tensorboard_directory', help='Directory to store the results of the tensorboard callback.', type=str, nargs=1, default=tbd_default, action=StoreSingleValueAction)
        sysconfig_group_opt.add_argument('-svcp', '--save_checkpoint', help='Whether to store the output of the checkpoint callback.', type=str, nargs=1, default='True', action=StoreSingleValueAction, choices=Session.save_options)
        cpd_default = f'{MODELS_PATH}\\{ID_MARKER}\\ckpts\\' if PLATFORM == 'WINDOWS' else f'{MODELS_PATH}/{ID_MARKER}/ckpts/'
        sysconfig_group_opt.add_argument('-cpd', '--checkpoint_directory', help='Directory to store the checkpoints created during training after each epoch.', type=str, nargs=1, default=cpd_default, action=StoreSingleValueAction)
        sysconfig_group_opt.add_argument('-svhis', '--save_history', help='Whether to store the history object returned by keras.Model.fit().', type=str, nargs=1, default='True', action=StoreSingleValueAction, choices=Session.save_options)
        hisd_default = f'{MODELS_PATH}\\{ID_MARKER}\\' if PLATFORM == 'WINDOWS' else f'{MODELS_PATH}/{ID_MARKER}/'
        sysconfig_group_opt.add_argument('-hisd', '--history_directory', help='Directory to store the history object returned by keras.Model.fit().', type=str, nargs=1, default=hisd_default, action=StoreSingleValueAction)
        sysconfig_group_opt.add_argument('-svm', '--save_model', help='Whether to store the trained Keras model.', type=str, nargs=1, default='True', action=StoreSingleValueAction, choices=Session.save_options)
        md_default = f'{MODELS_PATH}\\{ID_MARKER}\\keras-model\\' if PLATFORM == 'WINDOWS' else f'{MODELS_PATH}/{ID_MARKER}/keras-model/'
        sysconfig_group_opt.add_argument('-md', '--model_directory', help='Directory to store the Keras model to.', type=str, nargs=1, default=md_default, action=StoreSingleValueAction)


        # model group
        model_group_req = self.add_argument_group('required MODEL arguments')
        model_group_req.add_argument('-mc', '--model_class', help='The model class name to train, e. g. UnetPlusPlus. MUST be specified.', type=str, nargs=1, required=True, action=StoreSingleValueAction, choices=Session.model_classes.keys())
        model_group_req.add_argument('-mp', '--model_params', help='The model parameter version to train with, e. g. default. MUST be specified.', type=str, nargs=1, required=True, action=StoreSingleValueAction, choices=Session.get_model_param_names())
        
        model_group_opt = self.add_argument_group('optional MODEL arguments')
        

        # data group
        data_group_req = self.add_argument_group('required DATA arguments')
        
        data_group_opt = self.add_argument_group('optional DATA arguments')
        data_group_opt.add_argument('-bs', '--batch_size', help='Batch size during training.', type=int, default=32)
        data_group_opt.add_argument('-tg', '--training_generator', help='Version of the training batch generator.', type=str, default='default', choices=Session.batch_generators.keys())
        data_group_opt.add_argument('-vg', '--validation_generator', help='Version of the validation batch generator.', type=str, default='default', choices=Session.batch_generators.keys())
        data_group_opt.add_argument('-tts', '--train_test_split', help='The train-test-split to use on the batch generators.', type=float, nargs=1, default=0.75, action=StoreSingleValueAction)
        data_group_opt.add_argument('-utts', '--use_train_test_split', help='Whether to split the data using train_test_split or whether bg/tg have separate indices', type=str, nargs=1, default='True', action=StoreSingleValueAction, choices=Session.caching.keys())
        
        data_group_opt.add_argument('-aug', '--augmenter', help='The data augmenter version to use on the training generator.', type=str, nargs=1, default='default', action=StoreSingleValueAction, choices=Session.augmenters.keys())
        data_group_opt.add_argument('-c', '--caching', help='Whether to retrieve all data or rely on dynamic loading.', type=str, nargs=1, default='True', action=StoreSingleValueAction, choices=Session.caching.keys())
    

    def parse_args(self, args = None) -> argparse.Namespace:
        parsed: argparse.Namespace = super().parse_args(args=args)
        if len(str(parsed.id)) != 12:
            raise Exception('A session id must be an 12-digit integer.')
        return parsed