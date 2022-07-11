import tensorflow as tf
import numpy as np


def accuracy(y_pred: np.array, y_true: np.array) -> float:

    # binary activation with step function
    cond = tf.less(y_pred, 0.5)
    out = tf.where(cond, tf.zeros(tf.shape(y_pred)), tf.ones(tf.shape(y_pred)))
    y_pred = tf.cast(out,tf.int32)

    # cast ground truth 
    y_true = tf.cast(y_true,tf.int32)

    # calculate accuracy
    true_results = tf.cast(tf.equal(y_pred,y_true),tf.int32)
    true_results_count = tf.reduce_sum(true_results)
    all_count = tf.size(y_true)
    return tf.divide(true_results_count,all_count)