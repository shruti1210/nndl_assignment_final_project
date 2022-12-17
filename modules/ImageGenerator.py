import numpy as np
from matplotlib import pyplot as plt

class ImageGenerator(object):

    def __init__(self, x, y):
        #instance for imageGenerator
        # param x ---------- a numpy array for inputs, has a shape - number of sample, height, width and channels
        # param y ---------- a numpy vector for labels, shape ----- number of samples
        self.samples_count
        self.x = x
        self.y = y
        self.sample_count, self.height, self.width, self.channel_count = self.x.shape  # N,H,W,C
        # height shift
        self.height_shift = 0  
        # width shift
        self.width_shift = 0 
        # default  ----- no angle 
        self.angle = 0  
        # need a horizontal flip
        self.flip_horizontal = False  
        # need a vertical flip
        self.flip_vertical = False  
        #add noise or not
        self.add_noise = False

    def next_batch_gen(self, batchSize, shuffle=True):
        """
        a generator function to yield a data batch indefinitely 
        param 
            batch_size: number of samples to be returned for each batch
            shuffle: If True ----- shuffle entire dataset after every sample has been returned once
                        If False ----- order/data sample stays same
        :return: A batch of data with size ---- batch, size, width, height, channels
        """
        #total number of batches
        totalBatchNum = self.sampleCount
        batchNum = 0
        while True:
            if batchNum < totalBatchNum:
                batchNum += 1
                yield (self.x[(batchNum - 1) * batchSize : batchNum * batchSize],
                       self.y[(batchNum - 1) * batchSize : batchNum * batchSize])
            else:
                if shuffle:
                    permuations = np.random.permutation(self.samples_count)
                    self.x = self.x[permuations]
                    self.y = self.y[permuations]
                batchNum = 0

    def show(self):
        
        #Plot the top 16 images
        X_sample = self.x[:16]
        # imshow() one channel of images
        # dimension of plots matrix 4 * 4
        r = 4 
        # plot matrix and figure size
        f, axarr = plt.subplots(r, r, figsize=(8,8))  
        for i in range(r):
            for j in range(r):
                image = X_sample[r*i+j]
                axarr[i][j].imshow(image, cmap="gray")

    def shift(self, heightShift, widthShift):
        """
        translate self.x by height and width shift
        height_shift: pixels to shift along height direction, can be negative
        width_shift: pixels to shift along width direction, can be negative
        """
        self.height_shift = heightShift
        self.width_shift = widthShift

        #shift the values along axis 1 - height and axis 2 width, roll elements, 
        # if rolled later than the last position, re-introduce at the first position
        # shift in height, axis = 1, horizontal
        self.x = np.roll(self.x, heightShift, axis=1)
        # shift in width, axis = 2, vertical
        self.x = np.roll(self.x, widthShift, axis=2) 

    def flip(self, mode='h'):
        #flip self.x according to the specified mode, if h - horizontal, v - vertical
        # h = horizontal, default horizontal
        self.flip_horizontal = 'h' in mode  
        # v = vertical
        self.flip_vertical = 'v' in mode  
        # flip horizontally ----- flip on axis 2
        if self.flip_horizontal:
            self.x = np.flip(self.x, axis=2) 
        # flip vertically ---- flip upon axis 1 
        if self.flip_vertical:
            self.x = np.flip(self.x, axis=1) 