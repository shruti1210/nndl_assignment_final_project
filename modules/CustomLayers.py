from .SpectralPool import Common_Spectral_Pool
from .FrequencyDropout import freq_dropout_mask
import tensorflow as tf
import numpy as np



#Keras leveraged, create all layers here

class Spectral_Conv_Layer(tf.keras.layers.Layer):
    
    #Keras leveraged, define a Spectral Convolution Layer
    
    def __init__(self, filters, kernel_size=3, **kwargs):
        super(Spectral_Conv_Layer, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.kernel = None

    """ add trainable parameters to the layer -- build """

    def build(self, input_shape):
        self.sample_weight = self.add_weight(shape=(self.kernel_size, self.kernel_size),
                                 initializer=tf.keras.initializers.GlorotUniform(),
                                 trainable=True)
        self.bias = self.add_weight(shape=(self.kernel_size, self.kernel_size),
                                 initializer=tf.keras.initializers.GlorotUniform(),
                                 trainable=True)

    def call(self, input_tensor):
        complexSampleWeight = tf.cast(self.sample_weight, dtype=tf.complex64)
        """   find spectral weight using sample weight    """
        fft2dSampleWeight = tf.signal.fft2d(complexSampleWeight)
        realInit = tf.math.real(fft2dSampleWeight)
        imageInit = tf.math.imag(fft2dSampleWeight)
        spectralWeight = tf.complex(realInit, imageInit)
        """   convert spectral weights back to spatial, and take real of them    """
        complexSpatialWeight = tf.signal.ifft2d(spectralWeight)
        spatialWeight = tf.math.real(complexSpatialWeight)
        self.kernel = spatialWeight + self.bias
        self.kernel = tf.expand_dims(self.kernel, axis=-1, name=None)
        self.kernel = tf.keras.layers.Concatenate(axis=-1)([self.kernel for _ in range(input_tensor.shape[-1])])
        self.kernel = tf.expand_dims(self.kernel, axis=-1, name=None)
        self.kernel = tf.keras.layers.Concatenate(axis=-1)([self.kernel for _ in range(self.filters)])
        
        # Perform convolution using calculated spatial weight
        return tf.nn.conv2d(input=input_tensor, filters=self.kernel, strides=[1,1,1,1], padding='SAME')

class Spectral_Pool_Layer(tf.keras.layers.Layer):
    """
    Keras leveraged, Spectral Pooling Layer created via subclassing
    """
    def __init__(self, outChannels, kernel_size=3, frequencyDropoutLowerBound=None, frequencyDropoutUpperBound=None, **kwargs):
        super(Spectral_Pool_Layer, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.out_channels = outChannels
        self.training = True
        self.freq_dropout_lower_bound = frequencyDropoutLowerBound
        self.freq_dropout_upper_bound = frequencyDropoutUpperBound

    def call(self, input_tensor, activation=tf.nn.relu):
        """ 2d FFT """
        imgfft = tf.signal.fft2d(tf.cast(input_tensor, tf.complex64))

        """ to truncate the spectrum """
        imageTruncated = Common_Spectral_Pool(imgfft, self.kernel_size)
        """out with respect to the frequency drop out bound """
        if (self.freq_dropout_lower_bound is not None) and (self.freq_dropout_upper_bound is not None):
            """in case of training - drop frequencies above a randomly determined level"""
            if self.training:
                tf_random_cutoff = tf.random.uniform([], self.freq_dropout_lower_bound, self.freq_dropout_upper_bound)
                dropoutMask = freq_dropout_mask(self.kernel_size, tf_random_cutoff)
                dropoutMask = tf.expand_dims(dropoutMask, axis=-1, name=None)
                dropoutMask = tf.expand_dims(dropoutMask, axis=0, name=None)
                imageDownSampled = imageTruncated[:,:,:,:] * dropoutMask
            #during test, return unchanged truncated frequency matrix
            else:
                imageDownSampled = imageTruncated
            output = tf.math.real(tf.signal.ifft2d(imageDownSampled))
        else:
            output = tf.math.real(tf.signal.ifft2d(imageTruncated))

        if activation is not None:
            return activation(output)
        else:
            return output

def conv_2d_layer(filters, kernel_size):
    #Keras leveraged, create a Convoluntion2d layer via subclassing
    conv_2d = tf.keras.layers.Conv2D(
        filters,
        kernel_size,
        strides=(1, 1),
        padding="valid")
    return conv_2d(filters, kernel_size)

def Dense_layer(units):
    #Keras leveraged, create a Dense Layer
    Dense = tf.keras.layers.Dense(
        units,
        activation='relu',
        se_bias=tf.constant(True, dtype=tf.bool))
    return Dense(units)

def global_average_layer():
    #Keras leveraged, create a Global Average Pooling 2D Layer
    globalAverage = tf.keras.layers.GlobalAveragePooling2D()
    return globalAverage()

