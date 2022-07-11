import os
from data_generator.batch_generator_functions import convert_batches_to_ts, unstack_batches
import tensorflow as tf
import traceback
# tf.keras.utils.set_random_seed(42)
# tf.config.set_visible_devices([], 'GPU')
import numpy as np
from argparse import Namespace
import logging
import pickle
from time import sleep
from typing import Any, Callable, List, Tuple, Dict
import logging
import pandas as pd
import shutil

from model.loss_functions.dice_loss import dice_loss
from model.loss_functions.stepped_dice_loss import stepped_dice_loss
from model.loss_functions.tversky_loss import tversky_loss
from model.loss_functions.tf_cup_f1_score import tf_cup_f1_score_loop_based, tf_cup_f1_score_map_based
from model.metric_functions.metrics_config import MetricsConfig

from model.unet_plus_plus.unet_plus_plus import UnetPlusPlus
from model.unet_plus_plus.unet_plus_plus_params import UnetPlusPlusParams
from model.wen_keyes.wen_keyes import WenKeyes
from model.wen_keyes.wen_keyes_params import WenKeyesParams
from model.ueber_net.ueber_net import UeberNet
from model.ueber_net.ueber_net_params import UeberNetParams

from session.callbacks.learning_rate_scheduler import CustomLearningRateScheduler
from session_validation.callbacks.validation_callback import ValidationCallback

from data_generator.batch_generator import BatchGenerator
from data_generator.batch_generator_params import BatchGeneratorParams
from data_generator.data_augmenter import DataAugmenter

from utils.SYSCONFIG import DATA_PATH, ID_MARKER, PLATFORM, LOGS_PATH
from utils.util_functions import printc, get_updated
from utils.errors import NoSuchModelParametersError, UnmatchedLossParamError


class Session:
    
    losses = {
        'tf_cup_f1_score_loop_based' : lambda **_: tf_cup_f1_score_loop_based,
        'tf_cup_f1_score_map_based' : lambda **_: tf_cup_f1_score_map_based,
        'dice_loss' : lambda **_: dice_loss,
        #'stepped_dice_loss' : lambda **_: stepped_dice_loss,
        'tversky_loss' : lambda **kwargs: tversky_loss(beta=float(kwargs['beta'])),
        'BinaryCrossentropy' : lambda **_: tf.keras.losses.BinaryCrossentropy(),
    }

    # NOTE: for accepted kwargs refer to https://www.tensorflow.org/api_docs/python/tf/keras/optimizers
    optimizers = {
        'Adadelta' : lambda lr, **kwargs: tf.keras.optimizers.Adadelta(learning_rate=lr, **kwargs),
        'Adagrad' : lambda lr, **kwargs: tf.keras.optimizers.Adagrad(learning_rate=lr, **kwargs),
        'Adam' : lambda lr, **kwargs: tf.keras.optimizers.Adam(learning_rate=lr, **kwargs),
        'Adamax' : lambda lr, **kwargs: tf.keras.optimizers.Adamax(learning_rate=lr, **kwargs),
        'Ftrl' : lambda lr, **kwargs: tf.keras.optimizers.Ftrl(learning_rate=lr, **kwargs),
        'Nadam' : lambda lr, **kwargs: tf.keras.optimizers.Nadam(learning_rate=lr, **kwargs),
        'RMSprop' : lambda lr, **kwargs: tf.keras.optimizers.RMSprop(learning_rate=lr, **kwargs),
        'SGD' : lambda lr, **kwargs: tf.keras.optimizers.SGD(learning_rate=lr, **kwargs),        
    }

    model_classes = {
        'UnetPlusPlus' : UnetPlusPlus,
        'WenKeyes' : WenKeyes,
        'UeberNet' : UeberNet
    }

    model_param_classes = {
        'UnetPlusPlus' :  {
            'default' : UnetPlusPlusParams.default
        },
        'WenKeyes' : {
            'default' : WenKeyesParams.default,
            'v1' : WenKeyesParams.v1
        },
        'UeberNet' : {
            'default' : UeberNetParams.default,
            'best': UeberNetParams.best,
            'best2': UeberNetParams.best2,
            'best3': UeberNetParams.best3
        }
    }

    batch_generators = {
        'default' : BatchGeneratorParams.default,
        'augmented_v1' : BatchGeneratorParams.augmented_v1,
        'stepped_v1' : BatchGeneratorParams.stepped_v1,
        'tune_train' : BatchGeneratorParams.tune_train,
        'tune_val' : BatchGeneratorParams.tune_val,
        
        'submission_train_data' : BatchGeneratorParams.submission_train_data,
        'submission_val_data' : BatchGeneratorParams.submission_val_data,

        'post_processing_val' : BatchGeneratorParams.post_processing_val,
        'post_processing_train' : BatchGeneratorParams.post_processing_train,
    }

    augmenters = {
        'default' : None,
        'v1' : DataAugmenter.v1,
    }

    metrics = {
        'default' : MetricsConfig.default
    }

    true_false_dict = {
        'False' : False,
        'True' : True
    }

    shuffle = true_false_dict
    caching = true_false_dict
    save_options = true_false_dict

    @staticmethod
    def get_model_param_names():
        names = []
        for key in Session.model_param_classes.keys():
            names = [*names, *Session.model_param_classes[key]]
        return list(set(names))
    
    def __init__(self, args: Namespace, tune: bool = False, auto_init: bool = True) -> None:
        self.send_message('Initializing ...')
        
        self.tune = tune
        
        # training args
        self.id: int = args.id
        self.loss_params = self.parse_loss_args(args.loss_params)
        self.loss: Callable = Session.losses[args.loss] if self.loss_params == {} else Session.losses[args.loss](**self.loss_params)
        self.learning_rate: float = args.learning_rate
        self.epochs: int = args.epochs
        self.optimizer = Session.optimizers[args.optimizer](self.learning_rate)
        self.metrics = Session.metrics[args.metrics]
        self.shuffle: bool = Session.shuffle[args.shuffle]

        # sysconfig args
        self.processor_cores: int = args.processor_cores
        self.retries: int = args.retries
        self.csv_callback_file: str = self.parse_dir(args.csv_callback_file)
        self.log_file: str = self.parse_dir(args.log_file)
        if PLATFORM == 'WINDOWS':
            logging.basicConfig(filename=self.log_file, encoding='utf-8', level=logging.DEBUG)
        elif PLATFORM == 'UNIX':
            logging.basicConfig(filename=self.log_file, level=logging.DEBUG)

        self.save_tensorboard_callback: bool = Session.save_options[args.save_tensorboard_callback]
        self.tensorboard_directory: str = self.parse_dir(args.tensorboard_directory)
        self.save_checkpoint: bool = Session.save_options[args.save_checkpoint]
        self.checkpoint_directory: str = self.parse_dir(args.checkpoint_directory)
        self.save_history: bool = Session.save_options[args.save_history]
        self.history_directory: str = self.parse_dir(args.history_directory)
        self.save_model: bool = Session.save_options[args.save_model]
        self.model_directory: str = self.parse_dir(args.model_directory)

        # set config log file 
        self.config_log_file: str = self.parse_dir(args.config_log_file)

        # model args
        self.model_class = Session.model_classes[args.model_class]
        try:
            self.model_params = (Session.model_param_classes[args.model_class])[args.model_params]
        except KeyError:
            raise NoSuchModelParametersError(model_class=args.model_class, model_class_params=args.model_params)

        # data args
        self.batch_size: int = args.batch_size
        self.augmenter = Session.augmenters[args.augmenter]
        # might be obsolete all-together
        self.train_test_split: float = args.train_test_split

        # if the data should be splitted using the train_test_split or if bg_train and bg_val have other index-files
        self.use_train_test_split = Session.true_false_dict[args.use_train_test_split]

        # do not add train_test_split parameter if tune is enabled or use_train_test_split is False
        if self.tune or not self.use_train_test_split:
            self.training_generator_params: Dict[str, Any] = get_updated(params=Session.batch_generators[args.training_generator], augmenter=self.augmenter)
            self.validation_generator_params: Dict[str, Any] = get_updated(params=Session.batch_generators[args.validation_generator], validation_generator=True)
        else:
            self.training_generator_params: Dict[str, Any] = get_updated(params=Session.batch_generators[args.training_generator], train_test_split=self.train_test_split, augmenter=self.augmenter, batch_size=self.batch_size, shuffle=self.shuffle)
            self.validation_generator_params: Dict[str, Any] = get_updated(params=Session.batch_generators[args.validation_generator], train_test_split=self.train_test_split, validation_generator=True, batch_size=self.batch_size)

        self.caching: bool = args.caching

        self.initialized = False
        if auto_init:
            self.initialize()

        if not(self.tune):
            self._sync_logs_and_weights()
        self.send_message('Finished initialization.')

    def initialize(self, skip_generators = False) -> None:
        """Initializes the session. May be called automatically or manually. Loads and compiles the
        model, initializes the data generators (if needed), creates all required callbacks

        Args:
            skip_generators (bool, optional): if True initialize attempts to skip data generator initialization. Defaults to False.
        """

        if self.initialized:
            return

        self.send_message(message='Started initialize subroutine ...')
        # get model and compile it
        self.send_message(message=f'Retrieving model ...')
        self.model: tf.keras.Model = self.model_class(**self.model_params).get_model()
        self.send_message(message=f'Retrieved model with {self.model.count_params()} parameters successfully.')
        self.send_message(message=f'Compiling model ...')
        self.model.compile(
            optimizer=self.optimizer,
            loss=self.loss,
            metrics=self.metrics
        )
        self.send_message(message=f'Compiled model successfully.')

        # check whether the generators already exist to leverage
        # the advantage of resetting and not having to reload the data
        can_skip = False
        if skip_generators:
            self.send_message(message=f'Trying to skip generator initialization ...')
            try:
                self.train_generator
                self.validation_generator
                can_skip = True
            except AttributeError:
                self.send_message(message=f'Generators must be initialized to skip.')
                can_skip = False
        
        # initialize data generators
        if not(can_skip):
            self.send_message(message=f'Initializing data generators ...')
            self.train_generator = BatchGenerator(**self.training_generator_params)
            self.validation_generator = BatchGenerator(**self.validation_generator_params)
            if self.caching:
                self.X_train, self.y_train = self.train_generator.get_data()
                self.X_test, self.y_test = self.validation_generator.get_data()
            self.send_message(message=f'Initialized data generators.')

        # initialize callbacks
        csv_callback = tf.keras.callbacks.CSVLogger(self.csv_callback_file, append=True)
        self.callbacks = [
            csv_callback, 

            #ValidationCallback(session=self)
        ]
        if self.save_tensorboard_callback:
            tb_callback = tf.keras.callbacks.TensorBoard(log_dir=self.tensorboard_directory)
            self.callbacks.append(tb_callback)
        checkpoint_path = self.checkpoint_directory + 'cp-{epoch:04d}.ckpt'
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)
        self.callbacks.append(cp_callback)
        
        self.initialized = True
        self.send_message(message='Finished initialize subroutine.')

    def _sync_logs_and_weights(self):
        """Synchronizes the existing weights in a checkpoint directory with the respective logs in a log directory;
        synchronizes to the last point where both existed
        """

        self.send_message('Syncing existing training runs ...')
        self.send_message('Checking for previously run epoch logs and stored weights ...')
        
        # check status of CSV log
        if os.path.exists(self.csv_callback_file):
            df = pd.read_csv(self.csv_callback_file, index_col=False)
            self.latest_log: int = 0 if df.empty else df.shape[0]
            if self.latest_log == 0:
                self.send_message(f'Found log indicating no previous training (epoch: {self.latest_log}).')
            else:
                self.send_message(f'Found log indicating previous training until epoch {self.latest_log}.')
        else:
            self.latest_log = 0
        
        # check status of stored model checkpoints
        if os.path.exists(self.checkpoint_directory):
            self.latest_weights: str = tf.train.latest_checkpoint(self.checkpoint_directory)
            if not(self.latest_weights == None):
                latest_epoch_weights = int(self.latest_weights[-9:-5])
                self.send_message(message=f'Found weights from epoch {latest_epoch_weights} in {self.latest_weights}')
            else:
                latest_epoch_weights = 0
        else:
            latest_epoch_weights = 0
            os.makedirs(self.checkpoint_directory)
            specifier = ' temporary ' if not(self.save_checkpoint) else ' '
            self.send_message(message=f'Created{specifier}checkpoint directory {self.checkpoint_directory} ...')
        
        if self.latest_log > latest_epoch_weights:
            truncated_log = df[:latest_epoch_weights]
            self.latest_log = latest_epoch_weights
            truncated_log.to_csv(self.csv_callback_file, index=False)
        elif latest_epoch_weights > self.latest_log:
            self.latest_weights = self.latest_weights.replace(f'{latest_epoch_weights:04d}', f'{self.latest_log:04d}')
        self.send_message(message=f'Synced logs and weights at epoch {self.latest_log}.')
        
    def parse_dir(self, dir: str) -> str:
        if ID_MARKER in dir:
            return dir.replace(ID_MARKER, str(self.id))
        return dir

    def parse_loss_args(self, args: List[Any]) -> Dict[str, Any]:
        """parses the list of args as received and returned by the CLIParser

        Args:
            args (List[Any]): List of flag - value pairs

        Raises:
            UnmatchedLossParamError: thrown if a flag without a value/value without a flag is encountered

        Returns:
            Dict[str, Any]: Dict of flag - value pairs, e. g. {'beta': 0.4}
        """

        if len(args) == 0:
            return {}
        if len(args) // 2 < len(args) / 2:
            raise UnmatchedLossParamError(params=args)
        params = {}
        for i in range(0, len(args), 2):
            key = args[i]
            value = args[i+1]
            params[key] = value
        return params

    def send_message(self, message: str, **kwargs) -> None:
        printc(source='[Session]', message=message, **kwargs)
    
    def execute(self) -> tf.keras.callbacks.History:
        """executes a Session under the given params

        Raises:
            KeyboardInterrupt: rethrown to exit training early

        Returns:
            tf.keras.callbacks.History: History object containing the metrics
        """

        if self.latest_log == self.epochs:
            self.send_message('Training has already been completed. Loaded last version of model')
            self.model.load_weights(self.latest_weights).expect_partial()
            return None

        # define history as None to be able to return it even if no training occurred
        history = None

        try:
            retry = False
            synced = False

            while self.retries >= 1 and self.latest_log < self.epochs: 
                self.retries = self.retries - 1
                if retry:
                    if not(synced):
                        self._sync_logs_and_weights()
                        synced = True
                else:
                    retry = True

                try:
                    self.send_message(message=f'Starting run in current session with {self.retries} retries left and {self.epochs - self.latest_log} epochs remaining ...')

                    # load weights if they exist from a previous training session
                    if self.latest_log > 0:
                        self.send_message(message=f'Loading weights from epoch {self.latest_log} ...')
                        self.model.load_weights(self.latest_weights).expect_partial()
                        self.send_message(message=f'Successfully loaded weights from epoch {self.latest_log} ...')

                    remaining_epochs = self.epochs - self.latest_log
                    self.send_message(message=f'Started training of the model from epoch {self.latest_log} for {remaining_epochs} epochs ...')

                    # fit network
                    synced = False

                    if self.caching:
                        verbose = 0 if self.tune else 1
                        history: tf.keras.callbacks.History = self.model.fit(x=self.X_train, y=self.y_train, validation_data=(self.X_test, self.y_test), batch_size=self.train_generator.batch_size,
                            epochs=self.epochs, callbacks=self.callbacks, initial_epoch=self.latest_log, verbose=verbose)
                    else: 
                        history: tf.keras.callbacks.History = self.model.fit(x=self.train_generator, validation_data=self.validation_generator, 
                            epochs=self.epochs, callbacks=self.callbacks, use_multiprocessing=True, workers=self.processor_cores, 
                            initial_epoch=self.latest_log, verbose=verbose)
                    
                    if not(synced):
                        self._sync_logs_and_weights()
                        synced = True
                    
                    # try to save the current history object
                    if self.save_history:
                        hist_path = f'{self.history_directory}history-{self.latest_log:04d}.pkl'
                        self.send_message(message=f'Saving history object to {hist_path} ...')
                        try:
                            with open(hist_path, 'wb') as f:
                                pickle.dump(obj=history, file=f)
                            self.send_message(message=f'Successfully saved history object.')
                        except:
                            self.send_message(message=f'Failed to save history object.')
                except KeyboardInterrupt:
                    raise KeyboardInterrupt()
                except RuntimeError as r:
                    self.send_message(message=f'Caught RuntimeError during training. Trying to restart ...')
                    self.send_message(message=f'RuntimeError: {str(r)}') 
                    self.send_message(traceback.format_exc())
                    sleep(5.0)
                except Exception as e:
                    self.send_message(message=f'Caught exception during training. Trying to restart ...')
                    self.send_message(message=f'Exception: {str(e)}')
                    self.send_message(traceback.format_exc())

                    logging.exception(e, exc_info=True)
                    sleep(5.0)
        except KeyboardInterrupt:
            self.send_message(message='Stopped training early due to KeyboardInterrupt.')
        finally:
            self.terminate()
        return history
    
    def set_loss(self, loss_name: str, **kwargs):
        """sets the loss of the Session

        Args:
            loss_name (str): name of the loss function
        """
        self.loss = Session.losses[loss_name](**kwargs)


    def step(self, step_size=1) -> Dict[str, Any]:
        """trains the current model for step_size epochs; used for tuning;
        for training see self.execute()

        Args:
            step_size (int, optional): number of epochs to train the model for at once. Defaults to 1.

        Returns:
            Dict[str, Any]: History.history dictionary containing the training metrics
        """

        verbose = 0 if self.tune else 1
        history = self.model.fit(x=self.X_train, y=self.y_train, 
            validation_data=None, batch_size=self.train_generator.batch_size, 
            epochs=step_size, use_multiprocessing=True, workers=self.processor_cores,
            verbose=verbose)
        
        last_vals = {}
        for key, value in zip(history.history.keys(), history.history.values()):
            last_vals[key] = value[-1]
        
        return last_vals


    def get_predictions_as_ts(self) -> Tuple[List, List]:
        """gets the validation data, predicts values with the current model and 
        converts the predicted batches to time series of varying length using the available 
        batch information

        Returns:
            Tuple[List, List]: tuple of [predicted series, true series]
        """

        X_test, y_test = self.get_validation_data()
        y_pred = self.model.predict(X_test, 
            batch_size=self.validation_generator.batch_size,
            verbose=0, use_multiprocessing=True, workers=self.processor_cores)
        
        center = self.validation_generator.center
        center_offset = self.validation_generator.center_offset
        sample_freq = self.validation_generator.sample_freq

        batch_table = self.validation_generator.batches
        ignore_xsv = os.path.join(DATA_PATH, 'zero.xsv')

        y_pred_unstacked = unstack_batches(y_pred, batch_size=self.validation_generator.batch_size)
        y_test_unstacked = unstack_batches(y_test, batch_size=self.validation_generator.batch_size)

        pred_ts = convert_batches_to_ts(data=y_pred_unstacked, batch_table=batch_table, center=center, center_offset=center_offset, sample_freq=sample_freq,ignore=[ignore_xsv])
        true_ts = convert_batches_to_ts(data=y_test_unstacked, batch_table=batch_table, center=center, center_offset=center_offset, sample_freq=sample_freq, ignore=[ignore_xsv])

        return pred_ts, true_ts

    def get_training_prediction_as_ts(self) -> Tuple[List, List]:

        X_train, y_train = self.get_training_data()
        y_pred = self.model.predict(X_train, 
            batch_size=self.train_generator.batch_size,
            verbose=0, use_multiprocessing=True, workers=self.processor_cores)
        
        center = self.train_generator.center
        center_offset = self.train_generator.center_offset
        sample_freq = self.train_generator.sample_freq

        batch_table = self.train_generator.batches
        ignore_xsv = os.path.join(DATA_PATH, 'zero.xsv')

        y_pred_unstacked = unstack_batches(y_pred, batch_size=self.train_generator.batch_size)
        y_train_unstacked = unstack_batches(y_train, batch_size=self.train_generator.batch_size)

        pred_ts = convert_batches_to_ts(data=y_pred_unstacked, batch_table=batch_table, center=center, center_offset=center_offset, sample_freq=sample_freq,ignore=[ignore_xsv])
        true_ts = convert_batches_to_ts(data=y_train_unstacked, batch_table=batch_table, center=center, center_offset=center_offset, sample_freq=sample_freq, ignore=[ignore_xsv])

        return pred_ts, true_ts


    def get_validation_data(self) -> Tuple[np.array, np.array]:
        """loads returns the validation data two different ways depending 
        on the chosen setting to cache data or not

        Returns:
            Tuple[np.array, np.array]: tuple of [X_test, y_test]
        """

        if self.caching:
            X_test, y_test = self.X_test, self.y_test
        else:
            X_test, y_test = self.validation_generator.get_data()
        return X_test, y_test

    def get_training_data(self) -> Tuple[np.array, np.array]:
        """loads returns the validation data two different ways depending 
        on the chosen setting to cache data or not

        Returns:
            Tuple[np.array, np.array]: tuple of [X_test, y_test]
        """

        if self.caching:
            X_train, y_train = self.X_train, self.y_train
        else:
            X_train, y_train = self.train_generator.get_data()
        return X_train, y_train

    def load_keras_model(self, model_path: str) -> None:
        """loads the model from a provided path

        Args:
            model_path (str): path pointing to the keras model directory
        """

        del self.model
        self.model = tf.keras.models.load_model(model_path)

    def save_keras_model(self, model_path: str) -> str:
        """saves the model to a folder in the specified directory

        Args:
            model_path (str): directory to save the model folder to

        Returns:
            str: path the model was saved to
        """

        file_path = model_path + "/model"
        self.model.save(file_path)
        return file_path

    def reset(self, args: Namespace, model_params: Dict[str, Any], tune: bool = False, auto_init: bool = True, skip_generators: bool = True):
        """resets the config of the current session to avoid instantiating a new Session

        Args:
            args (Namespace): object returned by CLIParser, used to set different vars
            model_params (Dict[str, Any]): new params to configure the new model
            tune (bool, optional): Allows for resetting the tune flag. Usually set to True. Defaults to False.
            auto_init (bool, optional): Whether to call self.initialize automatically. If not initialize must be called manually before calling step, execute etc. Defaults to True.
        """
        self.send_message('Resetting session ...')
        self.initialized = False
        self.tune = tune
        self.auto_init = auto_init

        self.loss_params =  self.parse_loss_args(args.loss_params)
        self.loss: Callable = Session.losses[args.loss] if self.loss_params == {} else Session.losses[args.loss](**self.loss_params)
        self.learning_rate: float = args.learning_rate
        # self.optimizer ...

        del self.model
        self.model_params = model_params
        if self.auto_init:
            self.initialize(skip_generators=skip_generators)
        self.send_message('Reset session successfully ...')

    def terminate(self) -> None:
        self.send_message(message=f'Terminating session ...')
        if self.save_model:
            self.send_message(message=f'Saving model to {self.model_directory} ...')
            if not os.path.exists(self.model_directory):
                 os.makedirs(self.model_directory)
            self.model.save(self.model_directory)
            self.send_message(message=f'Successfully saved model.')
        
        if not(self.save_checkpoint) and not(self.save_model) and not(self.save_history):
            self.send_message(message=f'Attempting to remove temporary directory {self.history_directory} fully ...')
            shutil.rmtree(self.history_directory)
            self.send_message(message='Removed all temporary directories.')
        elif not(self.save_checkpoint):
            self.send_message(message='Attempting to remove temporary checkpoint directory ...')
            shutil.rmtree(self.checkpoint_directory)
            self.send_message(message='Removed temporary checkpoint directory.')
        self.send_message(message=f'Terminated session.')
