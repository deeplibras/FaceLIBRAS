# from __future__ import print_function
from keras.models import load_model
from matplotlib import pyplot

import datasets.Datasets as Datasets
import datetime
import Models
import os
import Utils

# Configure train
nb_epoch = 500
batch_size = 20
k = 10

# Set model shape
model_shape = (1, 100, 74)
nb_classes = 7

# Print the train start time
startTime = datetime.datetime.now()

for i in range(0,k):
	print('ITERATION ' + str(i))
	# Import model
	model, model_number = Models.loadModel08(nb_classes, model_shape)

	# Compile model
	model = Models.compileSGD(model)

	# Load the fold for this iteration
	X_train, Y_train, X_test, Y_test, nb_classes = Datasets.loadBatchKDEF(k, i)

	X_train = X_train.astype('float32')
	X_test = X_test.astype('float32')
	X_train /= 255
	X_test /= 255
	
	# Fit the model
	hist = model.fit(X_train, Y_train,
			batch_size=batch_size, 
			nb_epoch=nb_epoch,
			verbose=1, 
			validation_data=(X_test, Y_test),
			shuffle=True)

	# Save iteration history
	Utils.saveHistory('iteration - ' + srt(i), hist, model_number)

# Print the train end time
endTime = datetime.datetime.now()
print('Start Time: ' + str(startTime))
print('End Time: ' + str(endTime))

