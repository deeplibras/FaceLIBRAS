from keras.models import load_model
from matplotlib import pyplot

import datasets.Datasets as Datasets
import datetime
import Models
import os
import Utils

def facelibras(database_path, nb_epoch, use_augmentation = None, resize = None):
	if resize == None:
		resize = True

	if use_augmentation == None:
		use_augmentation = False

	# Configure train
	batch_size = 60
	k = 10

	# Set model shape
	model_shape = (1, 100, 100)
	nb_classes = 7

	#Load model
	model, model_number = Models.loadModel08(nb_classes, model_shape)

	# Compile model
	model = Models.compileSGD(model)

	# Load database
	X_train, Y_train, X_test, Y_test, nb_classes = Datasets.loadBatchs(k, 1, database_path, nb_classes, resize)

	X_train = X_train.astype('float32')
	X_test = X_test.astype('float32')
	X_train /= 255
	X_test /= 255
	
	# Fit the model
	if use_augmentation == False:
		hist = model.fit(X_train, Y_train,
				batch_size=batch_size, 
				nb_epoch=nb_epoch,
				verbose=1, 
				validation_data=(X_test, Y_test),
				shuffle=True)
	else:
		model, hist = Models.trainWithImageAugmentation(model, batch_size, nb_epoch, X_train, Y_train, X_test, Y_test)

	return hist, model
