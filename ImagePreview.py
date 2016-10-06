'''
Read a database and show the images with and without Keras image augmentation
'''

from matplotlib import pyplot
from keras.preprocessing.image import ImageDataGenerator

import numpy as np
import datasets.Datasets as Datasets

# Configure database
database   = 'DeepLIBRAS'
nb_classes = 7

# Read the data
X_train, Y_train, X_test, Y_test, nb_classes = Datasets.loadBatchs(10, 1, database, nb_classes)

# Show images with no data augmenation
print('Showing image with no data augmentation')
for i in range(0, 9):
	pyplot.subplot(330 + 1 + i)
	pyplot.imshow(X_test[i].reshape(100, 100), cmap=pyplot.get_cmap('gray'))
pyplot.show()

# Define data preparation
datagen = ImageDataGenerator(
	featurewise_center=False,  # set input mean to 0 over the dataset
	samplewise_center=False,  # set each sample mean to 0
	featurewise_std_normalization=False,  # divide inputs by std of the dataset
	samplewise_std_normalization=False,  # divide each input by its std
	zca_whitening=False,  # apply ZCA whitening
	rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
	width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
	height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
	horizontal_flip=True,  # randomly flip images
	vertical_flip=False)  # randomly flip images

# fit parameters from data
datagen.fit(X_train)

# Show images with data augmentation
print('Showing image with data augmentation')
for X_batch, y_batch in datagen.flow(X_train, Y_train, batch_size=9):
	for i in range(0, 9):
	    pyplot.subplot(330+1+i)
	    img = X_batch[i]
	    pyplot.imshow(img.reshape(100, 100), cmap=pyplot.get_cmap('gray'))
	pyplot.show()
	break
