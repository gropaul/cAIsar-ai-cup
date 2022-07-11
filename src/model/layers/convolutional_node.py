import math
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, BatchNormalization, Activation

class ConvolutionalNode(tf.keras.layers.Layer):
    """
    ConvolutionalNode inherits from the tf.keras.layers.Layer class.
    """

    def __init__(self, i: int, j: int, k: int, base_filter_count: int, kernel_size: int, strides: int, **kwargs):

        super().__init__(**kwargs)

        self.i = i
        self.j = j
        self.k = k
        self.base_filter_count = base_filter_count
        self.kernel_size = kernel_size
        self.strides = strides
        
        self.filters = self.base_filter_count * math.pow(2, self.i)
        self.conv_ijk = Conv1D(filters=self.filters, kernel_size=self.kernel_size, strides=self.strides, padding='same', name=f'Conv1D/X{self.i}{self.j}.{self.k}')
        self.batch_norm_ijk = BatchNormalization(name=f'BatchNorm/X{self.i}{self.j}.{self.k}')
        self.activation_ijk = Activation('relu', name=f'Activation/X{self.i}{self.j}.{self.k}')

    @tf.function
    def call(self, inputs, *args, **kwargs) -> tf.Tensor:
        x = self.conv_ijk(inputs)
        x = self.batch_norm_ijk(x)
        x = self.activation_ijk(x)

        return x
    
