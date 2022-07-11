import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Activation, BatchNormalization, Add, Multiply
from tensorflow.keras import backend as K
from tensorflow.image import ResizeMethod



class AdditiveAttentionGate(tf.keras.layers.Layer):
    """
    AdditiveAttentionGate inherits from the tf.keras.layers.Layer class.
    
    The implementation of the AdditiveAttentionGate is an adaption of the Attention Gate proposed in [1] and [2].
    The following implementation has been adapted for use with time-series.

    [1] O. Oktay, J. Schlemper, L. Le Folgoc, M. Lee, M. Heinrich, K. Misawa, K. Mori, S. McDonagh, N. Y Hammerla, B. Kainz, B. Glocker, and D. Rueckert, 
    "Attention U-Net: Learning where to look for the pancreas" 
    2018, arXiv:1804.03999. [Online]. Available: http://arxiv.org/abs/1804.03999 
    
    [2] J. Schlemper, O. Oktay, M. Schaap, M. Heinrich, B. Kainz, B. Glocker, and D. Rueckert, 
    "Attention gated networks: Learning to leverage salient regions in medical images"
    Med. Image Anal., vol. 53, pp. 197–207, Apr. 2019.
    """

    def __init__(self, intermediate_factor: float, x_res_kernel:int = 3, **kwargs):
        """init method of AdditiveAttentionGate

        Args:
            intermediate_factor (float): value between 0 and 1 to calculate intermediate F_int dimension of the relation W
            x_res_kernel (int, optional): Size of the x resampling kernel. Defaults to 3.
        """

        super().__init__(**kwargs)

        self.intermediate_factor = intermediate_factor
        self.x_res_kernel = x_res_kernel

    def build(self, input_shape):
        assert type(input_shape) is list
        assert len(input_shape) == 2, 'List of input tensors must be of length 2.'

        self.x_shape = input_shape[0]
        self.g_shape = input_shape[1]

        if self.x_shape[2] > self.g_shape[2]:
            F_int = int(self.intermediate_factor * (self.x_shape[2] - self.g_shape[2]) + self.g_shape[2])
        else:
            F_int = int(self.intermediate_factor * (self.g_shape[2] - self.x_shape[2]) + self.x_shape[2])
        
        # reshape gating signal to F_int filters
        # phi_g = W_g*g -> shape: (b, l_g, F_int)
        self.W_g = Conv1D(filters=F_int,
                      kernel_size=1,
                      strides=1,
                      padding='same',
                      name='Conv1D/phi_g')
        self.BN_phi_g = BatchNormalization(name='BN/phi_g')
        
        # resampling value (x) signal to shape of gating signal filters
        # theta_x = W_x*x -> shape: (b, l_g, F_int)
        self.W_x = Conv1D(filters=F_int,
                     kernel_size=self.x_res_kernel,
                     strides=(self.x_shape[1] // self.g_shape[1]),
                     padding='same',
                     name='Conv1D/theta_x')
        self.BN_theta_x = BatchNormalization(name='BN/theta_x')
        
        self.add_phi_theta = Add(name='Add/sum_phi_theta')
        self.act_of_sum = Activation('relu', name='Activation/sum_phi_theta')
        
        # calculate alpha as the sigmoid activation of psi * act_of_sum_xg [+ b_psi]
        self.psi = Conv1D(filters=1, kernel_size=1, padding='same', name='Conv1D/psi')
        self.act_alpha = Activation('sigmoid', name='Activation/alpha')
                
        # multiply x with attention map alpha
        self.gate = Multiply(name='Multiply/x_gated')

    @tf.function
    def call(self, inputs, *args, **kwargs) -> tf.Tensor:
        """call-function of the Layer

        Args:
            x (tf.Tensor): value tensor of the attention gate 
            g (tf.Tensor): gating signal of the attention gate

        Returns:
            tf.Tensor: gated x-value
        """

        x = inputs[0]
        g = inputs[1]
        
        
        # reshape gating signal to F_int filters
        # phi_g = W_g*g -> shape: (b, l_g, F_int)
        phi_g = self.W_g(g)
        phi_g = self.BN_phi_g(phi_g)
        
        # resampling value (x) signal to shape of gating signal filters
        # theta_x = W_x*x -> shape: (b, l_g, F_int)
        theta_x = self.W_x(x)
        theta_x = self.BN_theta_x(theta_x)
        
        sum_phi_theta = self.add_phi_theta([phi_g, theta_x])
        act_of_sum_xg = self.act_of_sum(sum_phi_theta)
        
        # calculate alpha as the sigmoid activation of psi * act_of_sum_xg [+ b_psi]
        psi_sum = self.psi(act_of_sum_xg)
        alpha = self.act_alpha(psi_sum)
        
        # upsample alpha -> shape: (bs, l, 1), therefore add and drop temporary "channel" dimension
        # and repeat vector along channel axis to match the shape of x
        reshaped_alpha = alpha[..., tf.newaxis]
        upsampled_alpha = tf.image.resize(reshaped_alpha, size=(self.x_shape[1] , 1), method=ResizeMethod.BILINEAR)
        repeated_alpha = K.repeat_elements(upsampled_alpha, self.x_shape[2], axis=2)
        repeated_alpha = K.squeeze(repeated_alpha, axis=3)
        
        # multiply x with attention map alpha
        x_gated = self.gate([x, repeated_alpha])
        
        return x_gated