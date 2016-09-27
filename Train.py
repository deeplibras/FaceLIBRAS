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
batch_size = 60
k = 10

# Set model shape
model_shape = (1, 100, 100)
database = 'KDEF'
nb_classes = 7

# Print the train start time
startTime = datetime.datetime.now()

for i in range(4,7):
	# Import model
	model, model_number = Models.loadModel01(nb_classes, model_shape)
	if i == 0:	
		model, model_number = Models.loadModel01(nb_classes, model_shape)
	if i == 1:	
		model, model_number = Models.loadModel02(nb_classes, model_shape)
	if i == 2:	
		model, model_number = Models.loadModel03(nb_classes, model_shape)
	if i == 3:	
		model, model_number = Models.loadModel04(nb_classes, model_shape)
	if i == 4:	
		model, model_number = Models.loadModel05(nb_classes, model_shape)
	if i == 5:	
		model, model_number = Models.loadModel07(nb_classes, model_shape)
	if i == 6:	
		model, model_number = Models.loadModel08(nb_classes, model_shape)

	print('Training model ' + str(model_number))

	# Compile model
	model = Models.compileSGD(model)

	# Load the fold for this iteration
	X_train, Y_train, X_test, Y_test, nb_classes = Datasets.loadBatchs(k, 1, database, nb_classes)

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
	Utils.saveHistory(database, hist, model_number)

# Print the train end time
endTime = datetime.datetime.now()
print('Start Time: ' + str(startTime))
print('End Time: ' + str(endTime))

