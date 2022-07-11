import math
from typing import Dict

from tensorflow.keras.layers import Input, Concatenate, BatchNormalization, Activation, MaxPool1D, Conv1D, UpSampling1D, Dropout
from tensorflow.keras.models import Model

'''
This network architecture follows the approach for univariate time series 
anomaly detection as presented by Tailai Wen and Roy Keyes in their paper

>> Time Series Anomaly Detection Using Convolutional Neural Networks 
and Transfer Learning <<
source: https://arxiv.org/pdf/1905.13628v1.pdf, accessed: 26.01.2022

Approximate network dimensions:
    Total params: 726,912
    Trainable params: 723,968
    Non-trainable params: 2,944
'''

class WenKeyes:

    def __init__(self, sequence_length: int, channels: int, class_count: int, 
        concat_axis: int, base_filter_count: int, kernel_size: Dict[str, int], pool_size: Dict[str, int], 
        block_count: int, strides: Dict[str, int], dropout: Dict[str, float], inputs = None) -> None:
        
        self.sequence_length = sequence_length
        self.channels = channels
        self.input_shape = (self.sequence_length, self.channels)
        
        self.class_count = class_count
        self.concat_axis = concat_axis
        self.base_filter_count = base_filter_count
        self.kernel_size = kernel_size
        
        self.pool_size = pool_size
        self.strides = strides
        self.block_count = block_count
        self.dropout = dropout

        self.inputs = inputs

        self._init_dict_args()
        
    
    def _init_dict_args(self):
        # initialize pool sizes
        for i in range(1, math.floor(self.block_count / 2) + 1):
            if str(i) in self.pool_size.keys():
                continue
            else:
                self.pool_size[str(i)] = self.pool_size['default']
        
        # initialize stride sizes
        for i in range(1, self.block_count + 1):
            conv_key_1 = f'{i}.1'
            conv_key_2 = f'{i}.2'
            for key in [conv_key_1, conv_key_2]:
                if key in self.strides.keys():
                    continue
                else:
                    self.strides[key] = self.strides['default']
        
        # initialize kernel sizes
        for i in range(1, self.block_count + 1):
            kernel_key_1 = f'{i}.1'
            kernel_key_2 = f'{i}.2'
            kernel_keys = [kernel_key_1, kernel_key_2]

            if (i >= math.floor(self.block_count / 2) + 1) and (i < self.block_count):
                kernel_keys.append(f'{i}.l')
            for key in kernel_keys:
                if key in self.kernel_size.keys():
                    continue
                else:
                    self.kernel_size[key] = self.kernel_size['default']
        
        # initialize dropout rates
        for i in range(1, self.block_count + 1):
            dropout_key = f'{i}'
            if dropout_key in self.dropout.keys():
                continue
            else:
                self.dropout[dropout_key] = self.dropout['default']
        
    def _conv_layer_block(self, inputs, block: int):
        '''
        creates one layer block of two convolutions with the 
        subsequent relu activation
        inputs: inputs to be fed into the layer
        block: index of the layer block in the model, indexing starts at 1
        '''

        names = [[None for _ in range(2)] for _ in range(3)]
        for i in range(2):
            names[0][i] = f'Conv1D/B{block}.LG{i}'
            names[1][i] = f'BatchNorm/B{block}.LG{i}'
            names[2][i] = f'Activation/B{block}.LG{i}'

        filter_block = block
        if block > (self.block_count / 2):
            filter_block = self.block_count - block + 1
        filters = self.base_filter_count * math.pow(2, filter_block - 1)

        conv1d_0 = Conv1D(filters=filters, kernel_size=int(self.kernel_size[f'{block}.1']), strides=self.strides[f'{block}.1'], padding='same', name=names[0][0])(inputs)
        batch_norm_0 = BatchNormalization(name=names[1][0])(conv1d_0)
        activation_0 = Activation('relu', name=names[2][0])(batch_norm_0)

        conv1d_1 = Conv1D(filters=filters, kernel_size=int(self.kernel_size[f'{block}.2']), strides=self.strides[f'{block}.2'], padding='same', name=names[0][1])(activation_0)
        batch_norm_1 = BatchNormalization(name=names[1][1])(conv1d_1)
        activation_1 = Activation('relu', name=names[2][1])(batch_norm_1)

        if self.dropout[str(block)] > 0.0:
            dropout = Dropout(self.dropout[str(block)], name=f'Dropout/B{block}')(activation_1)
            return dropout
        else:
            return activation_1


    def get_model(self) -> Model:
        inputs, outputs = self.get_core()
        model = Model(inputs, outputs, name='WEN_KEYES_U_Net')
        return model

    def get_core(self, skip_softmax=False) -> tuple:
        # core consist of 2k+1 layers
        # k on the way down and k on the way up with one tipping point

        if self.inputs == None:
            self.inputs = Input(shape=self.input_shape, name='Input/S')
        
        # encoding blocks
        conv_layer_outputs = {}
        max_pool_i = self.inputs
        for i in range(1, self.block_count // 2 + 1):
            inputs_i = max_pool_i
            
            conv_block_i = self._conv_layer_block(inputs=inputs_i, block=i)
            conv_layer_outputs[str(i)] = conv_block_i

            pool_size_i = self.pool_size[str(i)]
            max_pool_i = MaxPool1D(pool_size_i, padding='same', name=f'MaxPool1D/B{i}')(conv_block_i)

        # center block
        m = self.block_count // 2 + 1
        conv_block_m = self._conv_layer_block(inputs=max_pool_i, block=m)
        
        upsampling_rate_m = self.pool_size[str(m - 1)]
        upsampling_m = UpSampling1D(upsampling_rate_m, name=f'UpSampling1D/B{m}')(conv_block_m)
        post_up_conv_m = Conv1D(filters=self.base_filter_count * math.pow(2, self.block_count // 2 - 1), kernel_size=int(self.kernel_size[f'{m}.l']), padding='same', name=f'Conv1D/B{m}.L')(upsampling_m)

        # decoding blocks
        post_up_conv_i = post_up_conv_m
        for i, conv_index in zip(range(self.block_count // 2 + 2, self.block_count), range(self.block_count // 2, 1, -1)):
            corresp_conv_block = conv_layer_outputs[str(conv_index)]
            concat_i = Concatenate(axis=self.concat_axis, name=f'Concatenate/B{i}')([post_up_conv_i, corresp_conv_block])
            conv_block_i = self._conv_layer_block(concat_i, block=i)
            upsampling_rate_i = self.pool_size[str(conv_index - 1)]

            upsampling_i = UpSampling1D(upsampling_rate_i, name=f'UpSampling1D/B{i}')(conv_block_i)
            post_up_conv_i = Conv1D(filters=self.base_filter_count * math.pow(2, conv_index - 1), kernel_size=int(self.kernel_size[f'{i}.l']), padding='same', name=f'Conv1D/B{i}.L')(upsampling_i)

        concat_f = Concatenate(axis=self.concat_axis, name=f'Concatenate/B{self.block_count}')([post_up_conv_i, conv_layer_outputs[str(1)]])
        conv_block_f = self._conv_layer_block(concat_f, block=self.block_count)
        conv_layer_final = Conv1D(self.class_count, 1, name='Conv1D/BF')(conv_block_f)
        
        if skip_softmax:
            outputs = conv_layer_final
        else:
            outputs = Activation('sigmoid', name='Activation/BF')(conv_layer_final)

        return self.inputs, outputs
        

