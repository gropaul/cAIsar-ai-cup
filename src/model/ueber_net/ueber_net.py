from typing import Dict, List
from model.layers.additive_attention_gate import AdditiveAttentionGate
from model.layers.convolutional_node import ConvolutionalNode
from model.layers.inception_node import InceptionNode
from model.layers.upsampling_layer import UpsamplingLayer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Concatenate, BatchNormalization, Activation, MaxPool1D, Conv1D, Dropout, Lambda, Add, Reshape, Dense
import tensorflow.keras.backend as K

from utils.errors import TooFewFiltersError, UnknownArchitectureError


'''
This network architecture follows the different approaches presented in 
the paper

>> U-Net and Its Variants for Medical Image Segmentation: A Review of Theory and Applications <<
source: https://ieeexplore.ieee.org/document/9446143/, accessed: 05.04.2022

integreting them into a single model and adapting them for usage in time-series segmentation

Approximate network dimensions may vary greatly.
'''

class UeberNet:
    def __init__(self, sequence_length: int, channels: int, base_architecture: str, additional_architectures: List[str],
            backbone_length: int, base_filter_count: int, concat_axis: int, class_count: int, kernel_size: Dict[int, Dict[int, Dict[int, int]]], 
            pool_size: Dict[int, Dict[int, Dict[int, int]]], strides: Dict[int, Dict[int, Dict[int, int]]], dropout: Dict[int, Dict[int, Dict[int, float]]], 
            n_fold_convolutions: int, attention_kernel: Dict[int, Dict[int, int]], attention_intermediate: Dict[int, Dict[int, float]], 
            inception_kernel_size: Dict[int, Dict[int, Dict[int, int]]], meta_length: int, meta_dropout: float, post_dense_meta_dropout: Dict[int, Dict[int, float]], inputs = None) -> None:

        self.sequence_length = sequence_length
        self.meta_length = meta_length
        self.channels = channels
        self.ts_input_shape = (self.sequence_length, self.channels)

        # base_architecture in [U-net, U-net++]
        self.base_architecture = base_architecture
        if not(self.base_architecture in ['U-net', 'U-net++']):
            raise UnknownArchitectureError(architecture=self.base_architecture)
        
        # additional_architectures in ['Attention', 'Dense', 'Inception', 'Residual']
        self.additional_architectures = additional_architectures
        unknowns = []
        for arch in self.additional_architectures:
            if not(arch in ['Attention', 'Dense', 'Inception', 'Residual']):
                unknowns.append(arch)
        if not(len(unknowns) == 0):
            raise UnknownArchitectureError(architecture=str(unknowns))


        self.class_count = class_count
        self.concat_axis = concat_axis
        self.base_filter_count = base_filter_count
        if self.base_filter_count < 4 and 'Inception' in self.additional_architectures:
            error_message = f'Choosing the inception architecture requires a base_filter_count > 4:\nbase_filter_count provided {self.base_filter_count}'
            raise TooFewFiltersError(message=error_message)

        self.backbone_length = backbone_length
        self.i_max = backbone_length - 1
        self.j_max = backbone_length - 1

        self.pool_size = pool_size
        self.kernel_size = kernel_size
        self.strides = strides
        self.dropout = dropout

        self.n_fold_convolutions = n_fold_convolutions
        self.inception_kernel_size = inception_kernel_size
        self.attention_kernel = attention_kernel
        self.attention_intermediate = attention_intermediate
        self.meta_dropout = meta_dropout
        self.post_dense_meta_dropout = post_dense_meta_dropout

        # input layer
        self.inputs = inputs
        if self.inputs == None:
            self.ts_inputs = Input(shape=self.ts_input_shape, name='Input/TS')
        if self.meta_length > 0:
            meta_inputs = Input(shape=(self.meta_length,), name='Input/meta')
            # generalized dropout for meta data; different implementation commented below
            if self.meta_dropout > 0.0:
                self.meta_inputs = Dropout(self.meta_dropout, name=f'Dropout/meta')(meta_inputs)
            else:
                self.meta_inputs = meta_inputs
            self.inputs = [self.ts_inputs, meta_inputs]
        else:
            self.inputs = self.ts_inputs

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
                for k in range(0, self.n_fold_convolutions):
                    if not k in self.strides[i][j].keys():
                        self.strides[i][j][k] = self.strides[-1]
        
        # initialize kernel sizes and inception kernel sizes
        for j in range(0, self.j_max + 1):
            for i in range(0, self.i_max + 1 - j):
                if not(i in self.kernel_size.keys()):
                    self.kernel_size[i] = dict()
                if not(j in self.kernel_size[i].keys()):
                    self.kernel_size[i][j] = dict()
                for k in range(0, self.n_fold_convolutions):
                    if not k in self.kernel_size[i][j].keys():
                        self.kernel_size[i][j][k] = self.kernel_size[-1]
        for j in range(0, self.j_max + 1):
            for i in range(0, self.i_max + 1 - j):
                if not(i in self.inception_kernel_size.keys()):
                    self.inception_kernel_size[i] = dict()
                if not(j in self.inception_kernel_size[i].keys()):
                    self.inception_kernel_size[i][j] = dict()
                for k in range(0, self.n_fold_convolutions):
                    if not k in self.inception_kernel_size[i][j].keys():
                        self.inception_kernel_size[i][j][k] = self.inception_kernel_size[-1]
        
        
        # initialize dropout rates
        for j in range(0, self.j_max + 1):
            for i in range(0, self.i_max + 1 - j):
                if not(i in self.dropout.keys()):
                    self.dropout[i] = dict()
                if not(j in self.dropout[i].keys()):
                    self.dropout[i][j] = self.dropout[-1]
        
        # initialize attention_kernel and key_dims for attention
        for j in range(0, self.j_max + 1):
            for i in range(0, self.i_max + 1 - j):
                if not(i in self.attention_kernel.keys()):
                    self.attention_kernel[i] = dict()
                if not(j in self.attention_kernel[i].keys()):
                    self.attention_kernel[i][j] = self.attention_kernel[-1]
        for j in range(0, self.j_max + 1):
            for i in range(0, self.i_max + 1 - j):
                if not(i in self.attention_intermediate.keys()):
                    self.attention_intermediate[i] = dict()
                if not(j in self.attention_intermediate[i].keys()):
                    self.attention_intermediate[i][j] = self.attention_intermediate[-1]
        
        for j in range(0, self.j_max + 1):
            for i in range(0, self.i_max + 1 - j):
                if not(i in self.post_dense_meta_dropout.keys()):
                    self.post_dense_meta_dropout[i] = dict()
                if not(j in self.post_dense_meta_dropout[i].keys()):
                    self.post_dense_meta_dropout[i][j] = self.post_dense_meta_dropout[-1]


    def get_node(self, i: int, j: int):
        '''
        i: index along the downsampling layer
        j: index along the skip-pathway

        returns output of the node
        '''
        
        node_input = None

        if (i == 0) and (j == 0):
            # starting node (start of the backbone and the U-net++)
            if self.meta_length > 0:
                meta_inputs = self.meta_inputs

                # NOTE: possible implementation to vary dropout for the different meta data Dense-layers
                # if self.meta_dropout > 0.0:
                #     meta_inputs = Dropout(self.meta_dropout, name=f'Dropout/meta.{i}')(meta_inputs)

                dense_meta = Dense(self.sequence_length, name=f'Dense/meta{self.sequence_length}')(meta_inputs)
                if self.post_dense_meta_dropout[i][j] > 0.0:
                    dense_meta = Dropout(self.post_dense_meta_dropout[i][j], name=f'Dropout/meta{self.sequence_length}')(dense_meta)
                reshape_meta = Reshape((self.sequence_length, 1), name=f'Reshape/meta{self.sequence_length}')(dense_meta)
                node_input = Concatenate(name='Concatenate/Input')([self.ts_inputs, reshape_meta])
            else:
                node_input = self.ts_inputs
            node_input = BatchNormalization(name='BatchNorm/Input')(node_input)
        
        elif j == 0 and (i <= self.i_max):
            # if j == 0 the node is part of the backbone (encoder) -> normal U-net logic
            # nodes should contain the node X_{i-1, 0}
            pooling_ij = MaxPool1D(4, padding='same', name=f'MaxPool1D/X{i}{j}')(self.nodes[i-1][0])
            if self.meta_length > 0:
                neurons = int(self.sequence_length / (4 ** i))
                meta_inputs = self.meta_inputs

                # NOTE: possible implementation to vary dropout for the different meta data Dense-layers
                # if self.meta_dropout > 0.0:
                #     meta_inputs = Dropout(self.meta_dropout, name=f'Dropout/meta.{i}')(meta_inputs)
                dense_meta = Dense(neurons, name=f'Dense/meta{neurons}')(meta_inputs)
                if self.post_dense_meta_dropout[i][j] > 0.0:
                    dense_meta = Dropout(self.post_dense_meta_dropout[i][j], name=f'Dropout/meta{neurons}')(dense_meta)
                reshape_meta = Reshape((neurons, 1), name=f'Reshape/meta{neurons}')(dense_meta)
                node_input = Concatenate(name=f'Concatenate/meta{neurons}')([pooling_ij, reshape_meta])
            else:
                node_input = pooling_ij
                

        else:
            # j > 0 and i = any,
            # collect skip inputs for U-Net/U-Net++
            # apply Attention logic if applicable

            skip_inputs = [self.nodes[i][0]]
            if self.base_architecture == 'U-net++':
                skip_inputs = [*skip_inputs,  *[self.nodes[i][k] for k in range(1, j)]]

            upsampling_factor = int(K.int_shape(skip_inputs[0])[1] / K.int_shape(self.nodes[i+1][j-1])[1])
            assert K.int_shape(skip_inputs[0])[1] / K.int_shape(self.nodes[i+1][j-1])[1] == K.int_shape(skip_inputs[0])[1] // K.int_shape(self.nodes[i+1][j-1])[1], 'Upsampling factor must be calculable as integer.'
            upsampled = UpsamplingLayer(i, j, upsampling_factor, self.base_filter_count, self.kernel_size[i][j][0], self.strides[i][j][0], name=f'UpsamplingLayer/X{i+1}{j-1}')(self.nodes[i+1][j-1])

            concatenated = Concatenate(axis=self.concat_axis, name=f'Concatenate/X{i}{j}')([*skip_inputs, upsampled])
            if 'Attention' in self.additional_architectures:
                attention = AdditiveAttentionGate(intermediate_factor=self.attention_intermediate[i][j], x_res_kernel=self.attention_kernel[i][j], name=f'AdditiveAttentionGate/X{i}{j}')([concatenated, self.nodes[i+1][j-1]])
                concatenated = attention
            node_input = concatenated

        if 'Inception' in self.additional_architectures:
            activation_ij0 = InceptionNode(i, j, 0, self.base_filter_count, self.inception_kernel_size[i][j][0], self.concat_axis, name=f'InceptionNode/X{i}{j}.0')(node_input)
        else:
            activation_ij0 = ConvolutionalNode(i, j, 0, self.base_filter_count, self.kernel_size[i][j][0], self.strides[i][j][0], name=f'ConvolutionalNode/X{i}{j}.0')(node_input)


        # if Dense is activated store the activations within one node
        # to allow for skip-connections 
        if 'Dense' in self.additional_architectures:
            previous_layers = [activation_ij0]
        
        # if Residual store initial activation after one convolution
        # calculate the indexes to skip every 2nd/2 out of three convolutions 
        if 'Residual' in self.additional_architectures:
            activation_ij0 = Activation('relu', name=f'ActResidual/X{i}{j}.0')(activation_ij0)
            residual_skip = activation_ij0
            residual_indexes = [i for i in range(2, self.n_fold_convolutions - 1, 2)]
            if self.n_fold_convolutions % 2 == 1:
                residual_indexes = [*residual_indexes, self.n_fold_convolutions - 1]
            else:
                residual_indexes = [*residual_indexes[:-1], self.n_fold_convolutions - 1]
        
        activation_ijk = activation_ij0

        if self.n_fold_convolutions > 1:
            for k in range(1, self.n_fold_convolutions):
                if 'Dense' in self.additional_architectures:
                    # dense logic
                    activation_ijk = Concatenate(axis=self.concat_axis, name=f'ConcatDense/X{i}{j}.{k}')(previous_layers) if len(previous_layers) > 1 else activation_ijk
                
                if 'Inception' in self.additional_architectures:
                    activation_ijk = InceptionNode(i, j, k, self.base_filter_count, self.inception_kernel_size[i][j][k], self.concat_axis, name=f'InceptionNode/X{i}{j}.{k}')(activation_ijk)
                else:
                    activation_ijk = ConvolutionalNode(i, j, k, self.base_filter_count, self.kernel_size[i][j][k], self.strides[i][j][k], name=f'ConvolutionalNode/X{i}{j}.{k}')(activation_ijk)

                if 'Residual' in self.additional_architectures and (k in residual_indexes):
                    if 'Inception' in self.additional_architectures:
                        # the output of the previous inception block may be of a slightly larger dimension
                        # than the intended layer size, e. g. (512, 62) instead of (512, 61),
                        # this may lead to problems as adding requires matching shapes
                        if K.int_shape(activation_ijk)[2] != K.int_shape(residual_skip)[2]:
                            activation_ijk = Lambda(lambda x: x[:, :, 0:K.int_shape(residual_skip)[2]], name=f'LambdaResizeResInc/X{i}{j}.{k}')(activation_ijk)
                    activation_ijk = Add(name=f'AddResidual/X{i}{j}.{k}')([activation_ijk, residual_skip])
                    activation_ijk = Activation('relu', name=f'ActResidual/X{i}{j}.{k}')(activation_ijk)
                    residual_skip = activation_ijk                

                if 'Dense' in self.additional_architectures:
                    previous_layers.append(activation_ijk)
        
        if self.dropout[i][j] > 0.0:
            dropout = Dropout(self.dropout[i][j], name=f'Dropout/X{i}{j}')(activation_ijk)
            return dropout

        return activation_ijk

    def get_model(self) -> Model:
        inputs, outputs = self.get_core()
        model = Model(inputs, outputs, name='UeberNet')
        return model

    def get_core(self) -> tuple:
        for j in range(0, self.j_max + 1):
            for i in range(0, self.i_max + 1 - j):
                if not(i in self.nodes.keys()):
                    self.nodes[i] = dict()
                if self.base_architecture == 'U-net':
                    if j != 0 and ((j + i) + 1 != self.backbone_length):
                       continue
                self.nodes[i][j] = self.get_node(i=i, j=j)

        conv_layer_final = Conv1D(self.class_count, 1, name='Conv1D/F')(self.nodes[0][self.j_max])
        outputs = Activation('sigmoid', name='Activation/F')(conv_layer_final)

        return self.inputs, outputs