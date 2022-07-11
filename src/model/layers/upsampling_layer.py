import math
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, UpSampling1D

class UpsamplingLayer(tf.keras.layers.Layer):
    """
    UpsamplingLayer inherits from the tf.keras.layers.Layer class.
    """

    def __init__(self, i: int, j: int, upsampling_factor: int, base_filter_count: int, kernel_size: int, strides: int, **kwargs):

        super().__init__(**kwargs)

        self.i = i
        self.j = j
        self.upsampling_factor = upsampling_factor
        self.base_filter_count = base_filter_count
        self.kernel_size = kernel_size
        self.strides = strides

        self.upsampling = UpSampling1D(self.upsampling_factor, name=f'UpSampling1D/X{self.i+1}{self.j-1}')
        self.filters = self.base_filter_count * math.pow(2, self.i)
        self.post_up_conv = Conv1D(filters=self.filters, kernel_size=self.kernel_size, strides=self.strides, padding='same', name=f'Conv1D/X{self.i+1}{self.j-1}.U')

    @tf.function
    def call(self, inputs, *args, **kwargs) -> tf.Tensor:
        x = self.upsampling(inputs)
        x = self.post_up_conv(x)

        return x