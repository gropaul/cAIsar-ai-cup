import tensorflow as tf

def tversky_coefficient(beta: float = 0.5):
  def coefficient(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.math.sigmoid(y_pred)
    numerator = y_true * y_pred
    denominator = y_true * y_pred + beta * (1 - y_true) * y_pred + (1 - beta) * y_true * (1 - y_pred)

    return tf.reduce_sum(numerator) / tf.reduce_sum(denominator)

  return coefficient