import math
from typing import Dict

from tensorflow.keras.layers import Input, Concatenate, BatchNormalization, Activation, MaxPool1D, Conv1D, UpSampling1D, Dropout
from tensorflow.keras.models import Model

'''
This network architecture follows the approach presented in the paper

>> UNet++: A Nested U-Net Architecture for Medical Image Segmentation <<
source: http://arxiv.org/abs/1807.10165, accessed: 24.03.2022

and has been adapted for usage in time-series segmentation

Approximate network dimensions:
Single convolution:
    Total params: 420,082
    Trainable params: 418,258
    Non-trainable params: 1,824
Double convolution:
    Total params: 772,546
    Trainable params: 768,898
    Non-trainable params: 3,648
'''

class UnetPlusPlus:
    def __init__(self, sequence_length: int, channels: int, backbone_length: int, base_filter_count: int,
            concat_axis: int, class_count: int, kernel_size: Dict[int, Dict[int, int]], pool_size: Dict[int, Dict[int, int]], 
            strides: Dict[int, Dict[int, int]], dropout: Dict[int, Dict[int, float]], double_convolutions: bool,inputs = None) -> None:

        self.sequence_length = sequence_length
        self.channels = channels
        self.input_shape = (self.sequence_length, self.channels)

        self.class_count = class_count
        self.concat_axis = concat_axis
        self.base_filter_count = base_filter_count

        self.backbone_length = backbone_length
        self.i_max = backbone_length - 1
        self.j_max = backbone_length - 1

        self.pool_size = pool_size
        self.kernel_size = kernel_size
        self.strides = strides
        self.dropout = dropout

        self.double_convolutions = double_convolutions

        self.inputs = inputs
        self.nodes = dict()

        self._init_dict_args()

    def _init_dict_args(self):
        # initialize pool sizes
        # pooling only required along the backbone
        for i in range(0, self.i_max + 1):
            if not(i in self.pool_size.keys()):
                self.pool_size[i] = dict()
            if 0 in self.pool_size[i].keys():
                continue
            else:
                if not(0 in self.pool_size[i].keys()):
                    self.pool_size[i] = dict()
                self.pool_size[i][0] = self.pool_size[-1]
        
        # initialize stride sizes
        # required for all nodes
        for j in range(0, self.j_max + 1):
            for i in range(0, self.i_max + 1 - j):
                if not(i in self.strides.keys()):
                    self.strides[i] = dict()
                if not(j in self.strides[i].keys()):
                    self.strides[i][j] = dict()
                for k in range(0, 2):
                    if not k in self.strides[i][j].keys():
                        self.strides[i][j][k] = self.strides[-1]
        
        # initialize kernel sizes
        for j in range(0, self.j_max + 1):
            for i in range(0, self.i_max + 1 - j):
                if not(i in self.kernel_size.keys()):
                    self.kernel_size[i] = dict()
                if not(j in self.kernel_size[i].keys()):
                    self.kernel_size[i][j] = dict()
                for k in range(0, 2):
                    if not k in self.kernel_size[i][j].keys():
                        self.kernel_size[i][j][k] = self.kernel_size[-1]
        
        # initialize dropout rates
        for j in range(0, self.j_max + 1):
            for i in range(0, self.i_max + 1 - j):
                if not(i in self.dropout.keys()):
                    self.dropout[i] = dict()
                if not(j in self.dropout[i].keys()):
                    self.dropout[i][j] = self.dropout[-1]


    def get_node(self, i: int, j: int):
        '''
        i: index along the downsampling layer
        j: index along the skip-pathway
        nodes: nested dictonary of existing nodes

        returns output of the node
        '''
        
        node_input = None
        if (i == 0) and (j == 0):
            # starting node (start of the backbone and the U-net++)
            
            node_input = self.inputs
        
        elif j == 0 and (i <= self.i_max):
            # if j == 0 the node is part of the backbone (encoder) -> normal U-net logic
            # nodes should contain the node X_{i-1, 0}

            pooling_ij = MaxPool1D(4, padding='same', name=f'MaxPool1D/X{i}{j}')(self.nodes[i-1][0])
            node_input = pooling_ij

        else:
            # j > 0 and i = any,
            # U-net++ logic
            # nodes should contain the node X_{i+1, j-1}
    
            upsampled = UpSampling1D(4, name=f'UpSampling1D/X{i+1}{j-1}')(self.nodes[i+1][j-1])
            other_inputs = [self.nodes[i][k] for k in range(0, j)]
            concatenated = Concatenate(axis=self.concat_axis, name=f'Concatenate/X_{i}{j}')([*other_inputs, upsampled])
            node_input = concatenated

        filters = self.base_filter_count * math.pow(2, i)

        conv_ij = Conv1D(filters=filters, kernel_size=self.kernel_size[i][j][0], strides=self.strides[i][j][0], padding='same', name=f'Conv1D/X{i}{j}.0')(node_input)
        batch_norm_ij = BatchNormalization(name=f'BatchNorm/X{i}{j}.0')(conv_ij)
        activation_ij = Activation('relu', name=f'Activation/X{i}{j}.0')(batch_norm_ij)

        if self.double_convolutions:
            conv_ij = Conv1D(filters=filters, kernel_size=self.kernel_size[i][j][0], strides=self.strides[i][j][1], padding='same', name=f'Conv1D/X{i}{j}.1')(activation_ij)
            batch_norm_ij = BatchNormalization(name=f'BatchNorm/X{i}{j}.1')(conv_ij)
            activation_ij = Activation('relu', name=f'Activation/X{i}{j}.1')(batch_norm_ij)

        
        if self.dropout[i][j] > 0.0:
            dropout = Dropout(self.dropout[i][j], name=f'Dropout/X{i}{j}')(activation_ij)
            return dropout

        return activation_ij

    def get_model(self) -> Model:
        inputs, outputs = self.get_core()
        model = Model(inputs, outputs, name='U_Net_plus_plus')
        return model

    def get_core(self) -> tuple:
        # input layer
        if self.inputs == None:
            self.inputs = Input(shape=self.input_shape, name='Input/S')
        self.inputs = BatchNormalization(name='BatchNorm/Input')(self.inputs)

        for j in range(0, self.j_max + 1):
            for i in range(0, self.i_max + 1 - j):
                if not(i in self.nodes.keys()):
                    self.nodes[i] = dict()
                self.nodes[i][j] = self.get_node(i=i, j=j)

        conv_layer_final = Conv1D(self.class_count, 1, name='Conv1D/F')(self.nodes[0][self.j_max])
        outputs = Activation('sigmoid', name='Activation/F')(conv_layer_final)

        return self.inputs, outputs

