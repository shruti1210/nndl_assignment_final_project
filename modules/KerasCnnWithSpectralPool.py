from .CustomLayers import Spectral_Conv_Layer, Spectral_Pool_Layer
import tensorflow as tf

class CNN_Spectral_Pool(tf.keras.Model):
    #CNN with spectral pooling layers and convolution layer options
    def __init__(self,
                 M,
                 l2_norm,
                 num_classes=10,
                 alpha=0.3,
                 beta=0.15,
                 max_num_filters=288,
                 use_parameterization = True
                 ):
        super(CNN_Spectral_Pool, self).__init__()
        self.num_classes = num_classes
        # M - total pairs - convolution and spectral-pool layer-pairs
        self.M = M
        self.alpha = alpha
        self.beta = beta
        self.max_num_filters = max_num_filters
        self.parameterization_or_not = use_parameterization
        # layers used in custom keras Model
        self.Spectral_Conv_Layer = Spectral_Conv_Layer
        self.Spectral_Pool_Layer = Spectral_Pool_Layer
        self.conv2d = tf.keras.layers.Conv2D(filters=64, padding='same', kernel_size=3, activation="relu", trainable=True)
        self.flatten = tf.keras.layers.Flatten()
        self.dense1000 = tf.keras.layers.Dense(1000, activation='relu',activity_regularizer=tf.keras.regularizers.l2(l=l2_norm))
        self.dense100 = tf.keras.layers.Dense(100, activation='relu',activity_regularizer=tf.keras.regularizers.l2(l=l2_norm))
        self.dense10 = tf.keras.layers.Dense(10, activation='relu',activity_regularizer=tf.keras.regularizers.l2(l=l2_norm))
        self.global_avg = tf.keras.layers.GlobalAveragePooling2D()
        self.softmax = tf.keras.layers.Softmax()
        self.Spectral_Conv_Layer_list = []
        self.Spectral_Pool_Layer_list = []
        self.conv2d_list = []
        self.filter_size = 3
        # for m in (1,M) - spectral convolution layers and spectral pooling layers are different
        for m in range(1, self.M + 1):  
            frequencyDropoutLowerBound, frequencyDropoutUpperBound = self.Freq_Dropout_Bounds(self.filter_size, m)
            numOfFilters = self.Num_of_Filters(m)
            self.conv2d_list.append(
                tf.keras.layers.Conv2D(
                    filters=numOfFilters,
                    padding='same',
                    kernel_size=3,
                    activation="relu",
                    trainable=True))
            self.Spectral_Conv_Layer_list.append(
                self.Spectral_Conv_Layer(filters=10, name='Spectral_Conv_Layer{0}'.format(m), trainable=True))
            self.Spectral_Pool_Layer_list.append(
                self.Spectral_Pool_Layer(
                out_channels=10,
                freqDropoutLowerBound=frequencyDropoutLowerBound,
                frequencyDropoutUpperBound=frequencyDropoutUpperBound,
                name='Spectral_Pool_Layer{0}'.format(m)))


    def Num_of_Filters(self, m):
        #m - present layer number <6
        # return filter number <288
        return min(self.max_num_filters, 96 + 32 * m)

    def Freq_Dropout_Bounds(self, size, idx):
        #frequency dropout bounds
        #implement the linear parameterization of frequency dropout probabiltiy distribution
        # size - size of image in layer
        #idx - current layer index
        #return - frequency dropout lower bound for frequency dropoff and 
        #frequency dropout upper bound for frequency drop off
        
        c = self.alpha + (idx / self.M) * (self.beta - self.alpha)
        frequencyDropoutLowerBound = c * (1. + size // 2)
        frequencyDropoutUpperBound = (1. + size // 2)
        return frequencyDropoutLowerBound, frequencyDropoutUpperBound

    def call(self, input_tensor):
        # initial part of CNN model - pairs of convolution and spectral pooling layers
        # from m in range(1 to M)
        for m in range(self.M):
            # first layer - input shape as an argument
            if m == 0:  
                # x = self.conv2d_list[0](input_tensor) if not useSpectralParameterization
                # x = self.Spectral_Conv_Layer_list[0](input_tensor) if useSpectralParameterization
                if self.parameterization_or_not:
                    x = self.conv2d_list[0](input_tensor)
                else:
                    #spectral convolution layer
                    x = self.Spectral_Conv_Layer_list[0](input_tensor)
            #it is not the first layer        
            else: 
                # x = self.conv2d_list[m](input_tensor) if not useSpectralParameterization
                # x = self.Spectral_Conv_Layer_list[m](input_tensor) if useSpectralParameterization
                if self.parameterization_or_not:
                    x = self.conv2d_list[m](x)
                else:
                    #spectral convolution layer 
                    x = self.Spectral_Conv_Layer_list[m](x)
            # spectral_pool_layer is added
            x = self.Spectral_Pool_Layer_list[m](x)  
        x = self.flatten(input_tensor)
        x = self.dense1000(x)
        x = self.dense100(x)
        if self.num_classes == 10:
            x = self.dense10(x)
        x = self.softmax(x)

        return x

