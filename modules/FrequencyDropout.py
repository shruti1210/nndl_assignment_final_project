"""Implement a frequency dropout."""
import numpy as np
import tensorflow as tf

def freq_dropout_mask(size, truncateThreshold):
    
    #create a mask for frequency dropout
    """
        Args: 
            size ------------- integer, height of the image to create a mask for, 
                               for eg, 32*32 image, it is 32
            truncate_threshold ---------- scalar, tensor of shape (,)
            All frequencies above this are set to zero, an image with a height of 32, number above 16 has no effect.
            For an image with height 31, an input above 15 has no effect
        Returns:
            dropout_mask: Tensor of shape (height, height)
                result ---- multiply by an FFT ---- create a modified FFT where all frequencies above the 
                cutoff are set to zero. Therefore, value of mask = 1 for frequencies below truncate level, 
                0 for freqencies above it. Mask of values to retain, not to drop. 
        """
    truncateThresholdShape = truncateThreshold.get_shape().as_list()
    assert len(truncateThresholdShape) == 0

    halfLow = size // 2  # round down
    if size % 2 == 1:
        halfUp = halfLow + 1
    else:
        halfUp = halfLow

    indiceMask = np.concatenate((np.arange(halfUp) , np.arange(halfLow, 0, -1))).astype(np.float32)

    xSpread = np.broadcast_to(indiceMask, (size, size))
    ySpread = np.broadcast_to(np.expand_dims(indiceMask, -1), (size, size))
    highestFrequency = np.maximum(xSpread, ySpread)

    dropoutMask = tf.cast(tf.less_equal(highestFrequency, truncateThreshold), tf.complex64)

    return dropoutMask


def freq_dropout_test(images, truncateThreshold):
    # to show the use of _frequency_dropout_mask
    """
        Args: 
            images: n-d array of shape - number of images, height, width, number of channels
            truncateThreshold: Tensor of shape (,)  - scalar
            All frequencies above this are set to zero, an image with a height of 32, number above 16 has no effect.
            For an image with height 31, an input above 15 has no effect
        Returns:
            sample_images: n-d array of shape - number of images, height, width, number of channels
        """
    assert len(images.shape) == 4
    N, H, W, C = images.shape
    assert H == W

    frequencydpMask = freq_dropout_mask(H, truncateThreshold)

    tfImages = tf.constant(images, dtype=tf.complex64)
    tfImages = tf.squeeze(tfImages)

    if len(tfImages.shape)==2:
        fftImages = tf.signal.fft2d(tfImages)
        truncatedImages = tf.math.multiply(fftImages,frequencydpMask)
        sampleImages = tf.math.real(tf.signal.ifft2d(truncatedImages))

    if len(tfImages.shape)==3:
        fftImages1 = tf.signal.fft2d(tfImages[:,:,0])
        fftImages2 = tf.signal.fft2d(tfImages[:,:,1])
        fftImages3 = tf.signal.fft2d(tfImages[:,:,2])
        fftImages=[fftImages1,fftImages2,fftImages3]
        fftImages=np.moveaxis(fftImages, 0, -1)

        truncatedImages1 = tf.math.multiply(tf.squeeze(fftImages[:,:,0]),frequencydpMask)
        truncatedImages2 = tf.math.multiply(tf.squeeze(fftImages[:,:,1]),frequencydpMask)
        truncatedImages3 = tf.math.multiply(tf.squeeze(fftImages[:,:,2]),frequencydpMask)
        truncatedImages=[truncatedImages1,truncatedImages2,truncatedImages3]
        truncatedImages=np.moveaxis(truncatedImages, 0, -1)

        sampleImages1 = tf.math.real(tf.signal.ifft2d(truncatedImages[:,:,0]))
        sampleImages2 = tf.math.real(tf.signal.ifft2d(truncatedImages[:,:,1]))
        sampleImages3 = tf.math.real(tf.signal.ifft2d(truncatedImages[:,:,2]))
        sampleImages=[sampleImages1,sampleImages2,sampleImages3]
        sampleImages=np.moveaxis(sampleImages, 0, -1)
      
    if len(tfImages.shape)==4:
        
        tfImages = tf.experimental.numpy.moveaxis(tfImages, 3, 1)
        imagesfft = tf.signal.fft2d(tfImages)
        imagesTruncated = imagesfft * frequencydpMask
        imagesBack = tf.math.real(tf.signal.ifft2d(imagesTruncated)) 
        sampleImages = imagesBack
    return sampleImages