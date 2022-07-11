import tensorflow as tf

def stepped_dice_loss(y_true, y_pred):
    """
    y_true = tf.cast(y_true, tf.bool)
    y_pred = tf.math.sigmoid(y_pred)
    one = tf.constant([1])
    one = tf.cast(one,tf.float32)

    y_pred = tf.math.greater(y_pred, one) 

    equals = tf.math.equal(y_true, y_pred)
    equals = tf.cast(equals,tf.int16)

    all = tf.math.equal(y_true, y_true)
    all = tf.cast(all,tf.int16)
    all_sum = tf.reduce_sum(all)

    sum_true = tf.reduce_sum(equals)

    return 1 - sum_true / all_sum
    """

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.math.sig(y_pred)
    numerator = 2 * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred)

    return 1 - numerator / denominator