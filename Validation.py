# from __future__ import print_function
from keras.models import load_model
from matplotlib import pyplot

import datasets.Datasets as Datasets
import datetime
import Models
import os
import Utils

# Configure train
nb_epoch = 1
batch_size = 10
k = 2

# Set model shape
model_shape = (1, 100, 100)

for j in range(0,3):
	database   = None
	nb_classes = None

	if(j == 0):
		database = 'DeepLIBRAS'
		nb_classes = 6
	if(j == 1):
		database = 'CK+'
		nb_classes = 7
	if(j == 2):
		database = 'KDEF'
		nb_classes = 7

	startTime = datetime.datetime.now()

	for i in range(0,k):
		print('Fold ' + str(i + 1))
		print('Loading model')

		# Import model	
		model, model_number = Models.loadModel08(nb_classes, model_shape)

		# Compile model
		model = Models.compileSGD(model)
	
		print('Loading data')
		# Load the fold for this iteration
		X_train, Y_train, X_test, Y_test, nb_classes = Datasets.loadBatchs(k, i, database, nb_classes)

		X_train /= 255
		X_test /= 255
	
		print('Training')
		# Fit the model
		hist = model.fit(X_train, Y_train,
				batch_size=batch_size, 
				nb_epoch=nb_epoch,
				verbose=1, 
				validation_data=(X_test, Y_test),
				shuffle=True)

		# Save iteration history
		Utils.saveHistory(database+'_batch_' + str(i + 1), hist, model_number)

	endTime = datetime.datetime.now()
	Utils.saveTime(database + '\nStart Time: ' + str(startTime) + '\nEnd Time: ' + str(endTime))
