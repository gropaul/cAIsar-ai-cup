import tensorflow as tf

from model.wen_keyes.wen_keyes import WenKeyes
from model.wen_keyes.wen_keyes_params import WenKeyesParams
from model.unet_plus_plus.unet_plus_plus import UnetPlusPlus
from model.unet_plus_plus.unet_plus_plus_params import UnetPlusPlusParams
from model.ueber_net.ueber_net import UeberNet
from model.ueber_net.ueber_net_params import UeberNetParams


TYPE = 'UeberNet'        # ['U-Net', 'U-Net++', 'UeberNet']
PLOT = True
TEST = False

if TYPE == 'U-Net':
    model = WenKeyes(**WenKeyesParams.default).get_model()
    print(model.summary())
    if PLOT:
        tf.keras.utils.plot_model(model, to_file='.\\model.png', show_shapes=True)

elif TYPE == 'U-Net++':
    model = UnetPlusPlus(**UnetPlusPlusParams.default).get_model()
    print(model.summary())

    if PLOT:
        tf.keras.utils.plot_model(model, to_file='.\\model.png', show_shapes=True)

elif TYPE == 'UeberNet':
    default = {
                'sequence_length' : 1024, 
                'channels' : 16,
                'base_architecture' : 'U-net++',
                'additional_architectures' : ['Attention', 'Dense', 'Inception', 'Residual'],
                # 'additional_architectures' : ['Attention', 'Dense', 'Inception', 'Residual'],
                # 'additional_architectures' : ['Attention'],
                # 'additional_architectures' : [],

                'base_filter_count' : 16, 
                'backbone_length' : 4, 
                'concat_axis' : 2,

                'class_count' : 2, 

                'kernel_size' : {-1 : 7}, 
                'pool_size' : {-1 : 4},
                'dropout' : {-1 : 0.317842},
                'strides' : {-1 : 1},

                # n-fold convolutions
                'n_fold_convolutions' : 4,

                # Attention params
                'attention_kernel' : {-1 : 3},
                'attention_intermediate' : {-1 : 0.2},

                # Inception params
                'inception_kernel_size' : {-1 : 4},

                # meta params
                'meta_length' : 14,
                'meta_dropout' : 0.5,
                'post_dense_meta_dropout' : {-1: 0.5}
                }

    # default = UeberNetParams.best2
    model = UeberNet(**default).get_model()
    print(model.summary())
    if PLOT:
        tf.keras.utils.plot_model(model, to_file='.\\model.png', show_shapes=True)

    if TEST:
        for backbone_length in [2, 3, 4, 5, 6, 7]:
            for channels  in [4, 8, 16, 20]:
                for sequence_length  in [128, 256, 512, 1024, 2048]:
                    print(f'bb_len {backbone_length}, channels {channels}, seq_len {sequence_length}')
                    default = {
                        'sequence_length' : sequence_length, 
                        'channels' : 16,
                        'base_architecture' : 'U-net++',
                        # 'additional_architectures' : ['Attention', 'Dense', 'Inception', 'Residual'],
                        'additional_architectures' : ['Attention', 'Dense', 'Inception', 'Residual'],

                        'base_filter_count' : 21, 
                        'backbone_length' : backbone_length, 
                        'concat_axis' : 2,

                        'class_count' : 2, 

                        'kernel_size' : {-1 : 7}, 
                        'pool_size' : {-1 : 4},
                        'dropout' : {-1 : 0.317842},
                        'strides' : {-1 : 1},

                        # n-fold convolutions
                        'n_fold_convolutions' : 3,

                        # Attention params
                        'attention_kernel' : {-1 : 3},
                        'attention_intermediate' : {-1 : 0.2},

                        # Inception params
                        'inception_kernel_size' : {-1 : 4},

                        'meta_length' : 14,
                        'meta_dropout' : 0.5
                    }

                    model = UeberNet(**default).get_model()
                    # print(model.summary())
                    if PLOT:
                        tf.keras.utils.plot_model(model, to_file='.\\model.png', show_shapes=True)