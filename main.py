# from __future__ import print_function
from keras.models import load_model
import datasets.Datasets as Datasets
import datetime
import Models

# Configure train
nb_iteration = 1
nb_epoch = 100
batch_size = 32
USE_REALTIME_DATA_AUGUMENTATION = False

# Configure model import
importModelFrom  = '' # Use a empty string to not import a compiled model
model_number = 5

# Import dataset
X_train, Y_train, X_test, Y_test, nb_classes = Datasets.loadKDEF()

# Import or create the model
if(importModelFrom == ''):
	# Import model
	model, model_number = Models.loadModel08(nb_classes, X_train[0].shape)

	# Compile model
	model = Models.compileSGD(model)
else:
	model = load_model(importModelFrom)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# Summary model structure
model.summary()

# Print the train start time
print(datetime.datetime.now())

for i in range(0, nb_iteration):
	print('Iteration ' + str(i + 1) + "/" + str(nb_iteration))
	if(USE_REALTIME_DATA_AUGUMENTATION):
		model = Models.trainWithImageAugmentation(model, batch_size, nb_epoch, X_train, Y_train, X_test, Y_test)
	else:
		model = Models.train(model, batch_size, nb_epoch, X_train, Y_train, X_test, Y_test)
	
	# Test the model
	score = model.evaluate(X_test, Y_test, verbose=0)

	model.save('results/model'+str(model_number)+'_iteration'+str(i)+'_epoch'+str(nb_epoch)+'_acuracy'+str(score[1]))

# Test the model
score = model.evaluate(X_test, Y_test, verbose=0)

# Show results
print('Test score:', score[0])
print('Test accuracy:', score[1])

model.save('results/model'+str(model_number)+'_iteration'+str(i)+'_epoch'+str(nb_epoch)+'_acuracy'+str(score[1])+'FINAL')

# Print the train end time
print(datetime.datetime.now())

