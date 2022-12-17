import numpy as np
import tensorflow as tf
from .CustomLayers import Spectral_Conv_Layer
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, BatchNormalization, Input, Dropout, GlobalAveragePooling2D
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam


class CNN_Spectral_Param():
	"""
	This class builds and trains the generic and deep CNN architectures
	with and without spectral pooling to make comparison and derive the conclusion
	"""
	def __init__(self, numOut = 10, architect = 'generic', useSpectralParams = True, kernel_size = 3,
				 l2_norm = 0.001, filters = 128, learning_rate = 1e-6):


		#initialize parameters in class CNN_Spectral_Param()
		# numOut ---- number of classes to predict for output 
		# architect ---- architecture to build - deep/generic
		# useSpectralParams ---- flags to turn spectral parameterization on/off
		# kernel_size ---- size of convoluntional kernel	
		# 12_norm ---- 12 norm CNN weights scaling factor to calculate 12 loss
		# learning_rate ---- learning rate for Adam Optimizer
		# data_format ---- 'NHWC'/'NCHW' format for input images
		# random_seed ---- seed for initializers to create reproducable results 	 
		
		self.num_out = numOut
		self.architect = architect
		self.use_spectral_params = useSpectralParams
		self.kernel_size = kernel_size
		self.l2_norm = l2_norm
		self.filters = filters
		self.learning_rate = learning_rate


	def _build_generic_model(self, useSpectralParams):
		#build a generic architecture of CNN with and without spectal pooling. 
		# architecture - pair of convolution and maxpooling layers, 
		#spectral convolution and maxpooling layer
		#three fully-connected layers
		#a softmax or global averaging layers
		#useSpectralParams - uses if spectral convolution layers 
		#returns the tensorflow keras model
		
		if useSpectralParams == True:

			model = Sequential()
			# spectral convolution 1
			model.add(Spectral_Conv_Layer(self.filters,self.kernel_size, input_shape=(32, 32, 3)))
			# maxpool 1 
			model.add(tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
			# spectral convolution 2
			model.add(tf.keras.layers.Conv2D(filters=192, padding='same', kernel_size=self.kernel_size,
												   activation="relu", trainable=True))
			# maxpool 2
			model.add(tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
			# flatten
			model.add(tf.keras.layers.Flatten())
			# dense 1
			model.add(tf.keras.layers.Dense(1024, activation='relu',
										   activity_regularizer=tf.keras.regularizers.l2(l=self.l2_norm)))
			# dense 2
			model.add(tf.keras.layers.Dense(512, activation='relu',
										   activity_regularizer=tf.keras.regularizers.l2(l=self.l2_norm)))
			# dense 3
			model.add(tf.keras.layers.Dense(self.num_out, activation='relu',
										   activity_regularizer=tf.keras.regularizers.l2(l=self.l2_norm)))
			model.add(tf.keras.layers.Softmax())
			for layers in model.layers:
					layers.trainable = True
			optimizer = Adam(learning_rate=self.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-7)
# 			optimizer = tf.keras.optimizers.SGD(learning_rate=1e-4, momentum=0.0, nesterov=False)
			model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
			return model

		elif useSpectralParams == False:

			model = Sequential()
			# generic convolution 1
			model.add(tf.keras.layers.Conv2D(filters=96, padding='same', kernel_size=self.kernel_size,
												   activation="relu", trainable=True,input_shape=(32, 32, 3)))
			# maxpool 1
			model.add(tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
			# generic_conv2
			model.add(tf.keras.layers.Conv2D(filters=192, padding='same', kernel_size=self.kernel_size,
												   activation="relu", trainable=True))
			# maxpool 2
			model.add(tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
			# flatten
			model.add(tf.keras.layers.Flatten())
			# dense 1
			model.add(tf.keras.layers.Dense(1024, activation='relu',
											activity_regularizer=tf.keras.regularizers.l2(l=self.l2_norm)))
			# dense 2
			model.add(tf.keras.layers.Dense(512, activation='relu',
											activity_regularizer=tf.keras.regularizers.l2(l=self.l2_norm)))
			# dense 3
			model.add(tf.keras.layers.Dense(256, activation='relu',
											activity_regularizer=tf.keras.regularizers.l2(l=self.l2_norm)))
			model.add(tf.keras.layers.Dense(self.num_out, activation='softmax'))

			for layers in model.layers:
					layers.trainable = True
			optimizer = Adam(learning_rate=self.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-7)
			model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
			return model


	def _build_deep_model(self, useSpectralParams):
		#builds deep architecture of CNN with and without spectral convolution layer
		#architecture ----- 
		#back-to-back spectral convolutions, back-to-back spectral convolutions, max-pool
		#back-to-back convolutions, max-pool back-to-back 10-filter convoltions
		# and a global averaging layer
		#useSpectralParams - define use spectral convolution layers
		#returns the tensorflow keras model

		if useSpectralParams == True:

			model = Sequential()
			# spectral_convolution 1
			model.add(Spectral_Conv_Layer(self.filters-36, self.kernel_size, input_shape=(32, 32, 3)))
			# spectral_convolution 2
			model.add(tf.keras.layers.Conv2D(filters=96, padding='same', kernel_size=self.kernel_size, strides=(1, 1),
											 activation="relu", trainable=True))
			# spectral_convolution 3
			model.add(tf.keras.layers.Conv2D(filters=96, padding='same', kernel_size=self.kernel_size, strides=(1, 1),
											 activation="relu", trainable=True))
			# spectral_convolution 4
			model.add(Spectral_Conv_Layer(self.filters - 36, self.kernel_size))
			# maxpool1
			model.add(tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
			# spectral_convolution 5
			model.add(tf.keras.layers.Conv2D(filters=192, padding='same', kernel_size=self.kernel_size, strides=(1, 1),
											 activation="relu", trainable=True))
			# spectral_convolution 6
			model.add(tf.keras.layers.Conv2D(filters=192, padding='same', kernel_size=self.kernel_size, strides=(1, 1),
											 activation="relu", trainable=True))
			# maxpool 2
			model.add(tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
			# spectral convolution 7
			model.add(tf.keras.layers.Conv2D(filters= self.num_out, padding='same', kernel_size=self.kernel_size, strides=(1, 1),
											 activation="relu", trainable=True))
			# global average pooling
			model.add(tf.keras.layers.GlobalAveragePooling2D())
			for layers in model.layers:
					layers.trainable = True
			optimizer = Adam(learning_rate=self.learning_rate, decay=0.01)
			model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
			return model

		elif useSpectralParams == False:

			model = Sequential()
			# deep convolution 1
			model.add(tf.keras.layers.Conv2D(filters=96, padding='same', kernel_size=self.kernel_size, strides=(1, 1),
											 activation="relu", trainable=True, input_shape=(32, 32, 3)))
			# deep convolution 2
			model.add(tf.keras.layers.Conv2D(filters=96, padding='same', kernel_size=self.kernel_size, strides=(1, 1),
											 activation="relu", trainable=True))
			# # Batch normalization
			# model.add(BatchNormalization())
			# maxpool 1
			model.add(tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
			# deep convolution 3
			model.add(tf.keras.layers.Conv2D(filters=192, padding='same', kernel_size=self.kernel_size, strides=(1, 1),
											 activation="relu", trainable=True))
			# deep convolution 4
			model.add(tf.keras.layers.Conv2D(filters=192, padding='same', kernel_size=self.kernel_size, strides=(1, 1),
											 activation="relu", trainable=True))
			# maxpool 2
			model.add(tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
			# deep convolution 5
			model.add(tf.keras.layers.Conv2D(filters=10, padding='same', kernel_size=self.kernel_size,strides=(1, 1),
											 activation="relu", trainable=True))
			# global average pooling
			model.add(tf.keras.layers.GlobalAveragePooling2D())
			for layers in model.layers:
					layers.trainable = True
			optimizer = Adam(learning_rate=self.learning_rate, decay=0.01)
			model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
			return model