import math
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, MaxPool1D, Lambda, Concatenate
import tensorflow.keras.backend as K

class InceptionNode(tf.keras.layers.Layer):
    """
    InceptionNode inherits from the tf.keras.layers.Layer class.
    """

    def __init__(self, i: int, j: int, k: int, base_filter_count: int, kernel_size: int, concat_axis: int, **kwargs):
        super().__init__(**kwargs)

        self.i = i
        self.j = j
        self.k = k

        self.base_filter_count = base_filter_count
        self.kernel_size = kernel_size
        self.concat_axis = concat_axis

        self.filters = self.base_filter_count * math.pow(2, self.i)

        # inception logic 
        self.inception_filters = [int(self.filters // 4) for _ in range(0, 4)]
        for index in range(0, int(self.filters % 4)):
            self.inception_filters[index] += 1

        # calculate equivalents
        self.kernel_size_x3_equiv = self.kernel_size
        self.kernel_size_x1_equiv = int(self.kernel_size_x3_equiv * (1/3)) + 1
        self.kernel_size_x5_equiv = int(self.kernel_size_x3_equiv * (5/3))

        # add to x1 equivalent convolution
        self.inception_Ax1 = Conv1D(filters=self.inception_filters[0], kernel_size=self.kernel_size_x1_equiv, strides=1, padding='same', name=f'Conv1DInceptAx1/X{self.i}{self.j}.{self.k}')
                    
        # add to x3 equivalent convolution
        self.inception_Bx1 = Conv1D(filters=self.inception_filters[1], kernel_size=self.kernel_size_x1_equiv, strides=1, padding='same', name=f'Conv1DInceptBx1/X{self.i}{self.j}.{self.k}')
        self.inception_Bx3 = Conv1D(filters=self.inception_filters[1], kernel_size=self.kernel_size_x3_equiv, strides=1, padding='same', name=f'Conv1DInceptBx5/X{self.i}{self.j}.{self.k}')
                    
        # add to x5 equivalent convolution
        self.inception_Cx1 = Conv1D(filters=self.inception_filters[2], kernel_size=self.kernel_size_x1_equiv, strides=1, padding='same', name=f'Conv1DInceptCx1/X{self.i}{self.j}.{self.k}')
        self.inception_Cx5 = Conv1D(filters=self.inception_filters[2], kernel_size=self.kernel_size_x5_equiv, strides=1, padding='same', name=f'Conv1DInceptCx5/X{self.i}{self.j}.{self.k}')
                    
        # add pool and to x1 equivalent convolution
        self.inception_Dpool = MaxPool1D(4, padding='same', name=f'MaxPool1DInceptD/X{self.i}{self.j}.{self.k}')
        self.inception_Dx1 = Conv1D(filters=self.inception_filters[0], kernel_size=self.kernel_size_x1_equiv, strides=1, padding='same', name=f'Conv1DInceptDx1/X{self.i}{self.j}.{self.k}')
        
        self.concat_layer = Concatenate(axis=self.concat_axis, name=f'ConcatInception/X{self.i}{self.j}.{self.k}')
    
    def build(self, input_shape):
        self.tile_factor = 1
        
        x_Cx5_1 = input_shape[1]
        x_Dx1_1 = input_shape[1] // 4 if input_shape[1] > 4 else 1

        if x_Cx5_1 > x_Dx1_1:
            # calculate tile factor
            self.tile_factor = int(x_Cx5_1 / x_Dx1_1)
            assert x_Cx5_1 / x_Dx1_1 == x_Cx5_1 // x_Dx1_1, 'Tile factor must be calculable as integer.'
    
            self.lambda_tile = Lambda(tf.tile, arguments={'multiples':(1, self.tile_factor, 1)}, name=f'LambdaTileInceptD/X{self.i}{self.j}.{self.k}')
        
    @tf.function
    def call(self, inputs, *args, **kwargs) -> tf.Tensor:       
        # add to x1 equivalent convolution
        x_Ax1 = self.inception_Ax1(inputs)
                    
        # add to x3 equivalent convolution
        x_Bx1 = self.inception_Bx1(inputs)
        x_Bx3 = self.inception_Bx3(x_Bx1)
                    
        # add to x5 equivalent convolution
        x_Cx1 = self.inception_Cx1(inputs)
        x_Cx5 = self.inception_Cx5(x_Cx1)
                    
        # add pool and to x1 equivalent convolution
        x_Dpool = self.inception_Dpool(inputs)
        x_Dx1 = self.inception_Dx1(x_Dpool)
        if self.tile_factor != 1:
            x_Dx1 = self.lambda_tile(x_Dx1)
        
        # concatenate all inception layers
        inception_layers = [x_Ax1, x_Bx3, x_Cx5, x_Dx1]
        activation_ijk = self.concat_layer(inception_layers)
        
        return activation_ijk