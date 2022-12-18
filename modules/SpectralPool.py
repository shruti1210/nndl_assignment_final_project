import numpy as np
import tensorflow as tf

def Common_Spectral_Pool(pictures, filter_size):
    #ensure picture shapes in NHWC
    input_shape = pictures.get_shape().as_list()
    assert len(input_shape) == 4
    _, H, W, _ = input_shape
    assert H == W
    #input filter type - int
    assert type(filter_size) is int
    # input filter size >3
    assert filter_size >= 3
    
    # if filter size is even - 
    # divide picture into 9 parts - top left, top middle, top right, 
    # middle left, middle middle, middleright and
    # bottom left, bottom middle, bottom right

    if filter_size % 2 == 0:
        fs = filter_size / 2
        rootPointFive=pow(0.5,0.5)
        topLeft = pictures[:, :, :fs, :fs]

        topMiddle = \
            tf.expand_dims(tf.cast(rootPointFive, tf.complex64) * (pictures[:,:fs, fs,:] + pictures[:,:fs,-fs,:]), -1)

        topRight = pictures[:,:fs,-(fs-1):,:]

        middleLeft = \
            tf.expand_dims(tf.cast(rootPointFive, tf.complex64) * (pictures[:,fs,:fs,:] + pictures[:,-fs,:fs,:]),-2)

        middleMiddle = \
            tf.expand_dims(
                tf.expand_dims(
                    tf.cast(0.5, tf.complex64)*(
                            pictures[:,fs,fs,:] + pictures[:,fs,-fs,:] +pictures[:,-fs,fs,:] + pictures[:,-fs,-fs,:]),-1),
            -1
        )

        middleRight = \
            tf.expand_dims(tf.cast(rootPointFive, tf.complex64) * (pictures[:,fs,-(fs-1):,:] + pictures[:,-fs,-(fs-1):,:]),-2)

        bottomLeft = pictures[:,-(fs-1):,:fs,:]

        bottomMiddle = \
            tf.expand_dims(tf.cast(rootPointFive, tf.complex64) * (pictures[:,-(fs-1):,fs,:] + pictures[:,-(fs-1):,-fs,:]),-1)

        bottomRight = pictures[:,-(fs-1):,-(fs-1):,:]
        """Combine all separate 9 parts"""
        #NHWC ----- width axis
        topCombined = tf.concat([topLeft, topMiddle, topRight],axis=-2)
        #NHWC ----- width axis
        middleCombined = tf.concat([middleLeft, middleMiddle, middleRight],axis=-2)  
        # NHWC ----- width axis
        bottomCombined = tf.concat([bottomLeft, bottomMiddle, bottomRight],axis=-2)  
        # NHWC ----- height axis
        combineAll = tf.concat([topCombined, middleCombined, bottomCombined],axis=-3)  

    # if filter size is odd
    if filter_size % 2 == 1:
        fs = filter_size // 2
        #odd filter, divide picture into 4 parts, top left, top right, bottom left and bottom right
        topLeft = pictures[:,:fs+1,:fs+1,:]
        topRight = pictures[:,:fs+1,-fs:,:]
        bottomLeft = pictures[:,-fs:,:fs+1,:]
        bottomRight = pictures[:,-fs:,-fs:,:]
        #combine four separate parts 
        # NHWC at width axis
        topCombined = tf.concat([topLeft, topRight], axis=-2)  
         # NHWC at width axis
        bottomCombined = tf.concat([bottomLeft, bottomRight], axis=-2) 
        # NHWC at height axis
        combineAll = tf.concat([topCombined, bottomCombined], axis=-3)  

    return combineAll

#Shift the zero-frequency component to the center of the spectrum
# Fourier Shift
def tf_fftshift(matrix, n, axis=1):
    #perform function similar to numpy's fftshift
    # take images as a channel first numpy array of shape : (batch_size, height, width, channels)
    #fourier shift, no inverse, project shift on axis=1 of the spectrum
    cutPoint = (n + 1) // 2
    head = [0, 0, cutPoint, 0]
    tail = [-1, -1, cutPoint, -1]
    slice1 = tf.slice(matrix, head, [-1, -1, -1, -1])
    slice2 = tf.slice(matrix, [0, 0, 0, 0], tail)
    matrix_ = tf.concat([slice1, slice2], axis + 1)
    """Based on the matrix_ realize shift on axis=0 of the spectrum"""
    head = [0, cutPoint, 0, 0]
    tail = [-1, cutPoint, -1, -1]
    slice1 = tf.slice(matrix_, head, [-1, -1, -1, -1])
    slice2 = tf.slice(matrix_, [0, 0, 0, 0], tail)
    matrix__=tf.concat([slice1, slice2], axis + 1)

    return matrix__

# Inverse Fourier Shift
def tf_ifftshift(matrix, n, axis=1):
    #perform function similar to numpy's ifftshift
    # take images as a channel first numpy array of shape : (batch_size, channels, height, width)
    #fourier shift, no inverse, project shift on axis=1 of the spectrum
    cutPoint = n - (n + 1) // 2
    head = [0, 0, cutPoint, 0]
    tail = [-1, -1, cutPoint, -1]
    slice1 = tf.slice(matrix, head, [-1, -1, -1, -1])
    slice2 = tf.slice(matrix, [0, 0, 0, 0], tail)
    matrix_ = tf.concat([slice1, slice2], axis + 1)
    #based on matrix_ realize shift on axis=0 of the spectrum
    head = [0, cutPoint, 0, 0]
    tail = [-1, cutPoint, -1, -1]
    slice1 = tf.slice(matrix_, head, [-1, -1, -1, -1])
    slice2 = tf.slice(matrix_, [0, 0, 0, 0], tail)
    matrix__ = tf.concat([slice1, slice2], axis + 1)

    return matrix__

def spectral_pool(image, filter_size=3,
                  return_fft=False,
                  return_transformed=False,
                  ):
    #single spectral operation
    #Args - image - numpy image array, channels last, shape - batchsize, height, width, channels
    #               filter size - final dimension of filter required
    #               return_fft ---- boolean, if True, function also returns the raw fourier transform
    #returns image same as input
    imagefft = tf.signal.fft2d(tf.cast(image, tf.complex64))
    imageTransformed = Common_Spectral_Pool(imagefft, filter_size)
    #perform ishift and inverse fft, throw image part, compute inverse 2-D discrrete 
    # perform ishift and take the inverse fft and throw img part
    # Computes the inverse 2-dimensional discrete Fourier transform
    imageifft = tf.math.real(tf.signal.ifft2d(imageTransformed))
    # normalize image:
    channelMax = tf.math.reduce_max(input_tensor=imageifft, axis=(0, 1, 2))
    channelMin = tf.math.reduce_min(input_tensor=imageifft, axis=(0, 1, 2))
    imageOut = tf.math.divide(imageifft - channelMin,
                       channelMax - channelMin)
    #returns result of fft, returns raw ft
    if return_fft:
        return imagefft, imageOut
    #return result of fft
    elif return_transformed:
        return imageTransformed, imageOut
    #return result of fft
    else:
        return imageOut


def max_pool(image, pool_size=(2,2)):
    max_pool_2d = tf.keras.layers.MaxPooling2D(pool_size,strides=(1, 1), padding='valid')
    return max_pool_2d(image)

def max_pool_1(image, pool_size=(2,2)):
    max_pool_2d = tf.keras.layers.MaxPooling2D(pool_size,strides=(1, 1), padding='same')
    return max_pool_2d(image)


def l2_loss_images(originalImages, modImages):
    #calculate the loss of modified image vs original image, 12(originial-mod)/12(orig)
    #original_images - numpy array size - batch, dimensions
    #mod_images - numpy array of same dimension as original images
    #return loss
    n = originalImages.shape[0]
    # convert to 2d:
    originalImage = originalImages.reshape(n, -1)
    modImage = modImages.reshape(n, -1)
    # bring to same scale if the two scales not already
    if originalImage.max() > 2:
        originalImage = originalImage / 255.
    if modImage.max() > 2:
        modImage = modImage / 255.
    # perform normalization
    errorNorm = np.linalg.norm(originalImage - modImage, axis=0)
    baseNorm = np.linalg.norm(originalImage, axis=0)
    return np.mean(errorNorm / baseNorm)


def l2_loss_images_1(originalImages, modImages):
    #loss - modified bs original images, 12(originalImages-modImages)/12(originalImages)
    #original images - numpy array size batch, dimensions
    #mod images - numpy array of same dimensions as original images, return loss
    n = originalImages.shape[0]
    # convert to 2d:
    originalImage = originalImages.reshape(n, -1)
    modImage = tf.reshape(modImages,[n, -1])
    # bring to same scale
    if originalImage.max() > 2:
        originalImage = originalImage / 255.
    if tf.math.reduce_max(modImage) > 2:
        modImage = modImage / 255.
    #normalise
    error_norm = np.linalg.norm(originalImage - modImage, axis=0)
    base_norm = np.linalg.norm(originalImage, axis=0)
    return np.mean(error_norm / base_norm)