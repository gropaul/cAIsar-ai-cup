import numpy as np
import tensorflow as tf
from utils.tf_util_functions import tf_conversion, tf_printc
from utils.tf_util_functions import tf_convert_float_to_binary_mask, tf_convert_mask_to_cup_format

THRESHOLD_IoU = 0.75

@tf.function
def tf_cup_f1_score_loop_based(y_pred: tf.Tensor, y_true: tf.Tensor) -> tf.Tensor:
    with tf.device('/device:CPU:0'):
        @tf.function
        def process_channel(mask: tf.Tensor) -> list:
            binary_mask = tf_convert_float_to_binary_mask(mask)
            steps = tf_convert_mask_to_cup_format(binary_mask)
            return steps

        predictions = tf.ragged.constant([[[0, 0]]], dtype=tf.int32)
        ground_truth = tf.ragged.constant([[[0, 0]]], dtype=tf.int32)

        channels = y_pred[0].shape[1]
        for pred_index in tf.range(y_pred.shape[0]):
            tf.autograph.experimental.set_loop_options(
                shape_invariants=[
                    (predictions, tf.TensorShape([None, None, None])),
                    (ground_truth, tf.TensorShape([None, None, None]))]
                )

            mask_pred = y_pred[pred_index]
            mask_truth = y_true[pred_index]

            for channel_index in tf.range(channels):
                tf.autograph.experimental.set_loop_options(
                shape_invariants=[
                    (predictions, tf.TensorShape([None, None, None])),
                    (ground_truth, tf.TensorShape([None, None, None]))]
                )
                channel_pred = mask_pred[:, channel_index] if channels > 0 else mask_pred
                channel_truth = mask_truth[:, channel_index] if channels > 0 else mask_truth
                processed_pred = process_channel(mask=channel_pred)
                processed_truth = process_channel(mask=channel_truth)

                predictions = tf.concat((predictions, [processed_pred]), 0)
                ground_truth = tf.concat((ground_truth, [processed_truth]), 0)

        fscore = tf_fscore_step_detection(y_pred=predictions[1:], y_true=ground_truth[1:])
        return fscore 


@tf.function
def tf_cup_f1_score_map_based(y_pred: tf.Tensor, y_true: tf.Tensor) -> tf.Tensor:
    with tf.device('/device:CPU:0'):
        y_true_converted = tf_conversion(y_true)
        y_pred_converted = tf_conversion(y_pred)
        return_val = tf_fscore_step_detection(y_true_converted, y_pred_converted)
        return return_val

@tf.function
def tf_inter_over_union(interval_1, interval_2):
    a = interval_1[0]
    b = interval_1[1]
    c = interval_2[0]
    d = interval_2[1]

    intersection = tf.math.maximum(0, tf.math.minimum(b, d) - tf.math.maximum(a, c))
    if intersection > 0:
        union = tf.math.maximum(b, d) - tf.math.minimum(a, c)
    else:
        union = (b - a) + (d - c)
    
    return intersection / union

@tf.function
def _tf_step_detection_precision(step_list_true, step_list_pred):
    # NOTE: no working implementation existing
    # _check_step_list(step_list_pred)
    
    if step_list_pred.shape[0] == 0:  # empty prediction
        return 0.0

    n_correctly_predicted = 0
    detected_index_set = set()  # set of index of detected true steps
    
    for pred_index in tf.range(step_list_pred.nrows()):
        step_pred = step_list_pred[pred_index]
        for true_index in tf.range(step_list_true.nrows()):
            step_true = step_list_true[true_index]
            if (true_index.ref() not in detected_index_set) and (
                tf_inter_over_union(step_pred, step_true) > THRESHOLD_IoU
            ):
                n_correctly_predicted += 1
                detected_index_set.add(true_index.ref())
                break
                
    return n_correctly_predicted / tf.cast(step_list_pred.nrows(), dtype=tf.int32) 

@tf.function
def _tf_step_detection_recall(step_list_true, step_list_pred):
    # NOTE: no working implementation exists
    # _check_step_list(step_list_pred)

    n_detected_true = 0
    predicted_index_set = set()  # set of indexes of predicted steps
    
    for true_index in tf.range(step_list_true.nrows()):
        step_true = step_list_true[true_index]
        for pred_index in tf.range(step_list_pred.nrows()):
            step_pred = step_list_pred[pred_index]
            if (pred_index.ref() not in predicted_index_set) and (
                tf_inter_over_union(step_pred, step_true) > THRESHOLD_IoU
            ):
                n_detected_true += 1
                predicted_index_set.add(pred_index.ref())
                break
    recall = n_detected_true / tf.cast(step_list_true.nrows(), dtype=tf.int32)
    return recall

@tf.function(autograph=True)
def tf_fscore_step_detection(y_true: tf.Tensor, y_pred: tf.Tensor, return_all: bool = False) -> float:
    
    if y_true.shape[0] == 0:
        return 0.0
    
    fscore_list = tf.constant((0,), dtype=tf.float32)
    rec_list = tf.constant((0,), dtype=tf.float32)
    prec_list = tf.constant((0,), dtype=tf.float32)
    
    for index in tf.range(y_true.nrows()): 
        tf.autograph.experimental.set_loop_options(
            shape_invariants=[
                (fscore_list, tf.TensorShape([None])),
                (rec_list, tf.TensorShape([None])),
                (prec_list, tf.TensorShape([None]))] 
                    if return_all else
                [(fscore_list, tf.TensorShape([None]))]
        )

        step_list_true = y_true[index]
        step_list_pred = y_pred[index]
        prec = _tf_step_detection_precision(step_list_true, step_list_pred)
        rec = _tf_step_detection_recall(step_list_true, step_list_pred)
        
        if tf.math.is_nan(prec):
            prec = tf.cast(0.0, tf.float64)
        if tf.math.is_nan(rec):
            rec = tf.cast(0.0, tf.float64)
        
        if prec + rec < 1e-6:
            fscore = tf.cast(0.0, tf.float32)
        else:
            fscore = tf.cast((2 * prec * rec) / (prec + rec), tf.float32)

        fscore_list = tf.concat((fscore_list, [fscore]), 0)
        if return_all:
            rec_list = tf.concat((rec_list, [rec]), 0)
            prec_list = tf.concat((prec_list, [prec]), 0)
    
    if return_all:
        return tf.math.reduce_mean(fscore_list[1:]), tf.math.reduce_mean(prec_list[1:]), tf.math.reduce_mean(rec_list[1:])
    return tf.math.reduce_mean(fscore_list[1:])
    

@tf.function
def _check_step_list(step_list):
    
    def check_assertions(x):
        assert (tf.shape(x)[0] == 2)[0], f'A step consists of a start and an end: {x}.'
        assert x[0] < x[1]
        return x
    tf.map_fn(check_assertions, step_list)

    '''for step in step_list:
        assert len(step) == 2, f"A step consists of a start and an end: {step}."
        start, end = step
        assert start < end, f"start should be before end: {step}."'''
