from matplotlib import pyplot
from PIL import Image
from random import shuffle
from keras.preprocessing.image import ImageDataGenerator
from skimage import io, exposure, img_as_ubyte, img_as_float

import numpy as np

from ImageCutter import ImageCutter
from ImageCutter.utils import ImageUtils

# Load data
def readDatabaseCK():
	trainFile = 'CK+/train.txt'
	testFile = 'CK+/test.txt'

	X_train_unsuffle = []
	Y_train_unsuffle = []
	X_train = []
	Y_train = []

	# Open train info and split in lines
	with open(trainFile) as f:
		for line in f:
			# split file info
			info = line.split('@')

			# Read the image
			img = Image.open("CK+/"+info[0]).convert('L')

			# Append image to input list as a numpy array
			X_train_unsuffle.append(np.asarray(img).reshape(1, 100, 77))

			# Append category to category list
			Y_train_unsuffle.append(int(info[1]))

	# suffle train data
	index_shuf = range(len(X_train_unsuffle))
	shuffle(index_shuf)
	for i in index_shuf:
		X_train.append(X_train_unsuffle[i])
		Y_train.append(Y_train_unsuffle[i])

	X_test_unsuffle = []
	Y_test_unsuffle = []
	X_test = []
	Y_test = []

	# Open test info and split in lines
	with open(testFile) as f:
		for line in f:
			# split file info
			info = line.split('@')

			# Read the image
			img = Image.open("CK+/"+info[0]).convert('L')

			# Append image to input list as a numpy array
			X_test_unsuffle.append(np.asarray(img).reshape(1, 77, 100))

			# Append category to category list
			Y_test_unsuffle.append(int(info[1]))

	# suffle test data
	index_shuf = range(len(X_test_unsuffle))
	shuffle(index_shuf)
	for i in index_shuf:
		X_test.append(X_test_unsuffle[i])
		Y_test.append(Y_test_unsuffle[i])

	# Return data
	return np.array(X_train), np.array(Y_train), np.array(X_test), np.array(Y_test)

X_train, Y_train, X_test, Y_test = readDatabaseCK()
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
for i in range(0, len(X_train) ):
	X_train[i] = exposure.rescale_intensity(X_train[i], out_range='float')

# create a grid of 3x3 images
for i in range(0, 9):
	pyplot.subplot(330 + 1 + i)
	pyplot.imshow(X_test[i].reshape(77, 100), cmap=pyplot.get_cmap('gray'))
# show the plot
pyplot.show()

print(X_train.shape)

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

# create a grid of 3x3 images
for X_batch, y_batch in datagen.flow(X_train, Y_train, batch_size=9):
	for i in range(0, 9):
	    pyplot.subplot(330+1+i)
	    img = X_batch[i]
	    pyplot.imshow(img.reshape(77, 100), cmap=pyplot.get_cmap('gray'))
	# show the plot
	pyplot.show()
	break


# convert from int to float
#X_train = X_train.astype('float32')
#X_test = X_test.astype('float32')
'''
X_train, Y_train = readInputs("datasets/train.txt")
X_test, Y_test = readInputs("datasets/test.txt")

# Convert input to float32 and rescale color intensity
X_train = X_train.astype('float32')
for i in range(0, len(X_train) ):
	X_train[i] = exposure.rescale_intensity(X_train[i], out_range='float')
'''
