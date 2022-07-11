
import tensorflow as tf

from model.metric_functions.accuracy import accuracy
from model.loss_functions.dice_loss import dice_loss
from model.loss_functions.tversky_loss import tversky_loss

class MetricsConfig:

    default =[
        tf.keras.losses.BinaryCrossentropy(),
        dice_loss,
        tversky_loss(0.7),
        accuracy,
        tf.keras.metrics.AUC(curve='PR'), 
        tf.keras.metrics.Precision(),
        tf.keras.metrics.Recall(),
        tf.keras.metrics.PrecisionAtRecall(0.5),
        tf.keras.metrics.MeanAbsoluteError(),          
    ]