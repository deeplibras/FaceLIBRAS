from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot

def loadlModel01(nb_classes, input_shape):
	model = Sequential()
	model.add(Convolution2D(6, 3, 3, activation='relu', input_shape=input_shape, bias=True))
	model.add(Convolution2D(12, 3, 3, activation='relu', bias=True))
	model.add(Convolution2D(24, 3, 3, activation='relu', bias=True))
	model.add(MaxPooling2D((2,2), strides=(2,2)))
	model.add(Flatten())
	model.add(Dense(50, activation='relu', bias=True))
	model.add(Dropout(0.5))
	model.add(Dense(25, activation='relu', bias=True))
	model.add(Dropout(0.5))
	model.add(Dense(nb_classes, activation='softmax'))

	return model, 1

def loadlModel02(nb_classes, input_shape):
	model = Sequential()
	model.add(Convolution2D(6, 3, 3, activation='relu', input_shape=input_shape, bias=True))
	model.add(Convolution2D(12, 3, 3, activation='relu', bias=True))
	model.add(Convolution2D(24, 3, 3, activation='relu', bias=True))
	model.add(MaxPooling2D((2,2), strides=(2,2)))
	model.add(Flatten())
	model.add(Dense(50, activation='relu', bias=True))
	model.add(Dropout(0.5))
	model.add(Dense(25, activation='relu', bias=True))
	model.add(Dropout(0.5))
	model.add(Dense(nb_classes, activation='softmax'))

	return model, 2

def loadlModel03(nb_classes, input_shape):
	model = Sequential()
	model.add(Convolution2D(100, 5, 5, activation='tanh', input_shape=input_shape, bias=True))
	model.add(MaxPooling2D((2,2), strides=(2,2)))
	model.add(Convolution2D(100, 4, 4, activation='tanh', bias=True))
	model.add(MaxPooling2D((2,2), strides=(2,2)))
	model.add(Flatten())
	model.add(Dense(300, activation='tanh'))
	model.add(Dense(100, activation='tanh'))
	model.add(Dense(nb_classes, activation='softmax'))

	return model, 3

def loadlModel04(nb_classes, input_shape):
	model = Sequential()
	model.add(Convolution2D(5, 3, 3, border_mode='same', activation='tanh', input_shape=input_shape, bias=True))
	model.add(MaxPooling2D((2,2), strides=(2,2)))
	model.add(Dropout(0.25))
	model.add(Convolution2D(5, 3, 3, border_mode='same', activation='tanh', bias=True))
	model.add(MaxPooling2D((2,2), strides=(2,2)))
	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(150, activation='tanh'))
	model.add(Dropout(0.25))
	model.add(Dense(50, activation='tanh'))
	model.add(Dropout(0.25))
	model.add(Dense(nb_classes, activation='softmax'))

	return model, 4
	
def loadModel05(nb_classes, input_shape):
	model = Sequential()
	model.add(Convolution2D(100, 3, 3, border_mode='same', activation='tanh', input_shape=input_shape, bias=True))
	model.add(Convolution2D(100, 3, 3, border_mode='same', activation='tanh', bias=True))
	model.add(Convolution2D(100, 3, 3, border_mode='same', activation='tanh', bias=True))
	model.add(MaxPooling2D((2,2), strides=(2,2)))
	model.add(Dropout(0.25))
	model.add(Convolution2D(100, 3, 3, border_mode='same', activation='tanh', bias=True))
	model.add(Convolution2D(100, 3, 3, border_mode='same', activation='tanh', bias=True))
	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(150, activation='tanh'))
	model.add(Dropout(0.25))
	model.add(Dense(50, activation='tanh'))
	model.add(Dropout(0.25))
	model.add(Dense(nb_classes, activation='softmax'))

	return model, 5

def loadModel07(nb_classes, input_shape):
	model = Sequential()
	model.add(Convolution2D(100, 3, 3, border_mode='same', activation='tanh', input_shape=input_shape, bias=True))
	model.add(Convolution2D(100, 3, 3, border_mode='same', activation='tanh', bias=True))
	model.add(Convolution2D(100, 3, 3, border_mode='same', activation='tanh', bias=True))
	model.add(MaxPooling2D((2,2), strides=(2,2)))
	model.add(Dropout(0.25))
	model.add(Convolution2D(100, 3, 3, border_mode='same', activation='tanh', input_shape=input_shape, bias=True))
	model.add(Convolution2D(100, 3, 3, border_mode='same', activation='tanh', bias=True))
	model.add(Convolution2D(100, 3, 3, border_mode='same', activation='tanh', bias=True))
	model.add(MaxPooling2D((2,2), strides=(2,2)))
	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(150, activation='tanh'))
	model.add(Dropout(0.25))
	model.add(Dense(50, activation='tanh'))
	model.add(Dropout(0.25))
	model.add(Dense(nb_classes, activation='softmax'))

	return model, 7

def loadModel08(nb_classes, input_shape):
	model = Sequential()
	model.add(Convolution2D(15, 3, 3, border_mode='same', activation='tanh', input_shape=input_shape, bias=True))
	model.add(Convolution2D(30, 3, 3, border_mode='same', activation='tanh', bias=True))
	model.add(MaxPooling2D((2,2), strides=(2,2)))
	model.add(Dropout(0.25))
	model.add(Convolution2D(15, 3, 3, border_mode='same', activation='tanh', bias=True))
	model.add(Convolution2D(30, 3, 3, border_mode='same', activation='tanh', bias=True))
	model.add(MaxPooling2D((2,2), strides=(2,2)))
	model.add(Dropout(0.25))
	model.add(Convolution2D(15, 3, 3, border_mode='same', activation='tanh', bias=True))
	model.add(Convolution2D(30, 3, 3, border_mode='same', activation='tanh', bias=True))
	model.add(MaxPooling2D((2,2), strides=(2,2)))
	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(500, activation='tanh'))
	model.add(Dropout(0.25))
	model.add(Dense(250, activation='tanh'))
	model.add(Dropout(0.25))
	model.add(Dense(nb_classes, activation='softmax'))

	return model, 8

def compileSGD(model):
	sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(loss='categorical_crossentropy',
		          optimizer=sgd,
		          metrics=['accuracy'])

	return model

def compileAdaDelta(model):
	model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

	return model

def trainWithImageAugmentation(model, batch_size, nb_epoch, X_train, Y_train, X_test, Y_test):
	# Define data preparation
	datagen = ImageDataGenerator(
		featurewise_center=False,  # set input mean to 0 over the dataset
		samplewise_center=False,  # set each sample mean to 0
		featurewise_std_normalization=False,  # divide inputs by std of the dataset
		samplewise_std_normalization=False,  # divide each input by its std
		zca_whitening=False,  # apply ZCA whitening
		rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
		width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
		height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
		horizontal_flip=True,  # randomly flip images
		vertical_flip=False)  # randomly flip images

	# compute quantities required for featurewise normalization
	# (std, mean, and principal components if ZCA whitening is applied)
	datagen.fit(X_train)

	'''
	for X_batch, y_batch in datagen.flow(X_train, Y_train, batch_size=9):
		for i in range(0, 9):
		    pyplot.subplot(330+1+i)
		    img = X_batch[i]
		    pyplot.imshow(img.reshape(100, 74), cmap=pyplot.get_cmap('gray'))
		# show the plot
		pyplot.show()
		break
	'''

	# fits the model on batches with real-time data augmentation:
	model.fit_generator(datagen.flow(X_train, Y_train,
		                batch_size=batch_size),
		                samples_per_epoch=X_train.shape[0],
		                nb_epoch=nb_epoch,
		                validation_data=(X_test, Y_test))

	return model

def train(model, batch_size, nb_epoch, X_train, Y_train, X_test, Y_test):
	# Fit the model
	model.fit(X_train, Y_train,
		    batch_size=batch_size, 
		    nb_epoch=nb_epoch,
		    verbose=1, 
		    validation_data=(X_test, Y_test),
		    shuffle=True)

	return model
