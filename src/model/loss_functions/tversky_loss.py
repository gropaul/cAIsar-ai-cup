from typing import Callable
import tensorflow as tf

def tversky_loss(beta: float = 0.5, alpha: float = None) -> Callable:
  """Tversky loss function, for details refer to Notion Docs

  Args:
      beta (float, optional): beta. Defaults to 0.5.
      alpha (float, optional): alpha, if None alpha = 1 - beta. Defaults to None.

  Returns:
      Callable: returns a callable tversky loss function
  """
  
  if alpha == None:
    alpha = 1 - beta

  def loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.math.sigmoid(y_pred)
    numerator = y_true * y_pred
    denominator = y_true * y_pred + beta * (1 - y_true) * y_pred + alpha * y_true * (1 - y_pred)

    return 1 - tf.reduce_sum(numerator) / tf.reduce_sum(denominator)

  return loss