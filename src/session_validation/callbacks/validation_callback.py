



import pandas as pd
import os
import tensorflow as tf

from utils.SYSCONFIG import LOGS_PATH

from data_processing.post_processing.post_processing_functions import join_dataset_steps
from utils.util_functions import transform_predictions_to_ts
from cup_scripts.metric import fscore_step_detection

def get_validation_callback_data_path(session_id: int):
    return os.path.join(LOGS_PATH,str(session_id) + "_validation_callback_logs.csv")

class ValidationCallback(tf.keras.callbacks.Callback):

    def __init__(self, session, use_training: bool = False):
        super().__init__()

        self.session = session
        self.use_training = use_training
        self.df_path = get_validation_callback_data_path(session.id)

        if os.path.exists(self.df_path):
            self.df = pd.read_csv(self.df_path)
        else:
            self.df = pd.DataFrame()

        

    def on_epoch_end(self, epoch, logs=None):

        if self.use_training:
            pred_ts, true_ts = self.session.get_training_prediction_as_ts()
        else:
            pred_ts, true_ts = self.session.get_predictions_as_ts()

        y_pred = transform_predictions_to_ts(y_pred=pred_ts)
        y_true = transform_predictions_to_ts(y_pred=true_ts)

        f_score, precision, recall = fscore_step_detection(y_true, y_pred)

        joined_y_pred = join_dataset_steps(y_pred)
        joined_f_score, joined_precision, joined_recall = fscore_step_detection(y_true, joined_y_pred)

        new_row = {
            'epoch': [epoch], 

            'f_score': [f_score], 
            'precision': [precision], 
            'recall': [recall],

            'joined_f_score': [joined_f_score],
            'joined_precision': [joined_precision],
            'joined_recall': [joined_recall],
        }

        df2 = pd.DataFrame(new_row)
        self.df = pd.concat([self.df, df2], ignore_index = True, axis = 0)
        self.df.to_csv(self.df_path, columns=self.df.columns, index=False)
