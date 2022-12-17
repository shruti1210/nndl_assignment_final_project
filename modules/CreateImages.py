"""Create scaled and shifted images for exploration."""
from PIL import Image
import os
import numpy as np
# The path we store all the pictures
__DEFAULT_PATH = '/home/ecbm4040/Spectral_Representation_for_CNN/Images'


def openImage(file_name, path=__DEFAULT_PATH):
    """
    Opening images using Pillow
    open() ----- to read or write files 
    os module ---- for accessing the filesystem
    path parameters ------ passed as strings or bytes 
    """
    if file_name is None:
        raise ValueError('required filename')
    fullpath = os.path.join(path, file_name)
    """
    PIL image is represented using a class with same name in Image Module
    Image module also provied with factory functions - for eg function to create new images and to load images from the files
    """
    image = Image.open(fullpath).convert('RGBA')
    return image


def saveDerivedImage(image, file_name=None, path=__DEFAULT_PATH):
    #save the pillow images as png
    if file_name is None:
        # define the filename
        file_name = 'Derived/{0:08x}.png'.format(np.random.randint(2 ** 31))
    fullpath = os.path.join(path, file_name)
    """
    Python's OS module ----- functions to interact with operating systems
    OS - Python's standard utility module ----- a portable way to use operating system dependent functionality
    """
    os.makedirs(os.path.dirname(fullpath), exist_ok=True)
    image.save(fullpath, 'PNG')


def downscaleImage(originalImage, maxWidth, maxHeight):
    #rescale image to a smaller image
    originalWidth = originalImage.width
    originalHeight = originalImage.height

    #scaling factor / multiplying factor for existing dimensions
    widthMultiple = maxWidth / originalWidth
    heightMultiple = maxHeight / originalHeight
    multipleFactor = min(heightMultiple, widthMultiple)

    # Generate new images
    updatedWidth = int(originalWidth * multipleFactor)
    updatedHeight = int(originalHeight * multipleFactor)
    """
    two applications of a mathematical formula - Lanczos filters and Lanczos sampling 
    ----- Lanczos Filters ------- used as ---- low-pass filter / used to smoothly interpolate the digital signal values between the samples
    ----- Lanczos resampling ------ maps each sample of the given signal ---- translated and scaled copy ---- of Lanczos kernel 
    ----- Lanczos kernel - sinc function windowed by the central lobe of a second, longer sinc function
    ----- Sum of these translated and scaled kernels, evaluated at desired points
    """
    updatedImage = originalImage.resize((updatedWidth, updatedHeight),resample=Image.LANCZOS)

    return updatedImage


def addToBackground(
    foregroundImage,
    destinationLeft,
    destinationTop,
    destinationMaxWidth,
    destinationMaxHeight,
    backgroundImage=None,
    backgroundWidth=128,
    backgroundHeight=128,
):
    
    #Add image to image set in jupyter notebook
    #If backgroundImage == None, function creates a solid grey background image of dimension - backgroundWidth and backgroundHeight and copy image into it

    if backgroundImage is None:

        """
        PIL.Image.new() method creates an updated image with provided mode and size
        size ----- (width, height)-tuple, in pixels
        the color - single value for single-band images; 
        and a tuple for multi-band images ( with one value corresponding to each band )
        """
        updatedBackgroundImage = Image.new('RGBA',(backgroundWidth, backgroundHeight),'#7f7f7f')
    else:
        # Copy a part of the image 
        updatedBackgroundImage = backgroundImage.copy()

    rescaledForegroundImage = downscaleImage(foregroundImage,destinationMaxWidth,destinationMaxHeight,)
    updatedBackgroundImage.paste(rescaledForegroundImage,box=(destinationLeft, destinationTop),mask=rescaledForegroundImage)

    return updatedBackgroundImage


def makeRandomSize(destinationWidth=128, destinationHeight=128):
    """
    scale and a new location for the images
    """
    scalingFactor = np.random.randint(16,1 + min(destinationWidth, destinationHeight))
    leftValue = np.random.randint(0, 1 + destinationWidth - scalingFactor)
    topValue = np.random.randint(0, 1 + destinationHeight - scalingFactor)
    widthValue = scalingFactor
    heightValue = scalingFactor

    return leftValue, topValue, widthValue, heightValue
