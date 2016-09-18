from PIL import Image
from random import shuffle
from keras.utils import np_utils

import numpy as np
import os

# Load DeepLIBRAS image
def loadDeepLIBRAS():
	trainFile = os.path.join(os.path.dirname(__file__), 'DeepLIBRAS/train.txt')
	testFile = os.path.join(os.path.dirname(__file__), 'DeepLIBRAS/test.txt')

	# Image config
	original_width = 640
	original_height = 480
	cut_range = 50
	
	X_train_unsuffle = []
	Y_train_unsuffle = []
	X_train = []
	Y_train = []

	with open(trainFile) as f:
		for line in f:
			# Split the line to get file path, category and crop coords
			info = line.split('@')
			# Split the coords
			coords = info[2].split(',')

			# Read and crop the image
			img = Image.open(os.path.join(os.path.dirname(__file__), 'DeepLIBRAS/expressoes_faciais_jpg/'+info[0]+".jpg")).convert("L")
			img = ImageCutter.cut(img, {"x":int(coords[0]), "y":int(coords[1])}, cut_range)

			# Append image to input list as a numpy array
			X_train_unsuffle.append(np.asarray(img).reshape(1, 100, 100))
			# Append category to category list
			Y_train_unsuffle.append(int(info[1]))

	index_shuf = range(len(X_train_unsuffle))
	shuffle(index_shuf)
	for i in index_shuf:
		X_train.append(X_train_unsuffle[i])
		Y_train.append(Y_train_unsuffle[i])

	X_test_unsuffle = []
	Y_test_unsuffle = []
	X_test = []
	Y_test = []

	with open(testFile) as f:
		for line in f:
			# Split the line to get file path, category and crop coords
			info = line.split('@')
			# Split the coords
			coords = info[2].split(',')

			# Read and crop the image
			img = Image.open(os.path.join(os.path.dirname(__file__), 'DeepLIBRAS/expressoes_faciais_jpg/'+info[0]+".jpg")).convert("L")
			img = ImageCutter.cut(img, {"x":int(coords[0]), "y":int(coords[1])}, cut_range)

			# Append image to input list as a numpy array
			X_test_unsuffle.append(np.asarray(img).reshape(1, 100, 100))
			# Append category to category list
			Y_test_unsuffle.append(int(info[1]))

	index_shuf = range(len(X_test_unsuffle))
	shuffle(index_shuf)
	for i in index_shuf:
		X_test.append(X_test_unsuffle[i])
		Y_test.append(Y_test_unsuffle[i])

	# Return data
	return np.array(X_train), np.array(Y_train), np.array(X_test), np.array(Y_test)

# Load Yale Faces B image
def loadYaleFaces():
    subject = "subject"
    actions = ["happy", "normal", "sad", "surprised", "wink"]

    X_train_unsuffle = []
    Y_train_unsuffle = []
    X_test = []
    Y_test = []

    # Get train inputs
    for x in range(1,8):
        for y in range (0,5):
            img = Image.open(os.path.join(os.path.dirname(__file__), 'yalefaces/cropped/subject'+str(x).zfill(2)+"."+actions[y])).convert("L")
            X_train_unsuffle.append(np.array(img).reshape(1, 37, 49))
            Y_train_unsuffle.append(y)

    # Get train inputs
    for x in range(9,16):
        for y in range (0,5):
            img = Image.open(os.path.join(os.path.dirname(__file__), 'yalefaces/cropped/subject'+str(x).zfill(2)+"."+actions[y])).convert("L")
            X_test.append(np.array(img).reshape(1, 37, 49))
            Y_test.append(y)

    # Shuffle data
    X_train = []
    Y_train = []
    index_shuf = range(len(X_train_unsuffle))
    shuffle(index_shuf)
    for i in index_shuf:
        X_train.append(X_train_unsuffle[i])
        Y_train.append(Y_train_unsuffle[i])
    
    # Return data
    return np.array(X_train), np.array(Y_train), np.array(X_test), np.array(Y_test)

# Load the KDEF images
def loadKDEF():
	trainFile = os.path.join(os.path.dirname(__file__), 'KDEF/train.txt')
	testFile = os.path.join(os.path.dirname(__file__), 'KDEF/test.txt')
	nb_classes = 7

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
			img = Image.open(os.path.join(os.path.dirname(__file__), 'KDEF/'+info[0])).convert('L')

			# Append image to input list as a numpy array
			X_train_unsuffle.append(np.asarray(img).reshape(1, 100, 74))

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
			img = Image.open("datasets/KDEF/"+info[0]).convert('L')

			# Append image to input list as a numpy array
			X_test_unsuffle.append(np.asarray(img).reshape(1, 100, 74))

			# Append category to category list
			Y_test_unsuffle.append(int(info[1]))

	# suffle test data
	index_shuf = range(len(X_test_unsuffle))
	shuffle(index_shuf)
	for i in index_shuf:
		X_test.append(X_test_unsuffle[i])
		Y_test.append(Y_test_unsuffle[i])

	# Convert class to categorical
	Y_train = np_utils.to_categorical(np.array(Y_train), nb_classes)
	Y_test = np_utils.to_categorical(np.array(Y_test), nb_classes)

	# Return data
	return np.array(X_train), Y_train, np.array(X_test), Y_test, nb_classes

def loadKDEFinRGB():
	trainFile = os.path.join(os.path.dirname(__file__), 'KDEF/train.txt')
	testFile = os.path.join(os.path.dirname(__file__), 'KDEF/test.txt')
	nb_classes = 7

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
			img = Image.open(os.path.join(os.path.dirname(__file__), 'KDEF/'+info[0]))

			# Append image to input list as a numpy array
			X_train_unsuffle.append(np.asarray(img).reshape(3, 100, 74))

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
			img = Image.open("datasets/KDEF/"+info[0])

			# Append image to input list as a numpy array
			X_test_unsuffle.append(np.asarray(img).reshape(3, 100, 74)) # FORMATO (CANAIS, ALTURA, LARGURA)

			# Append category to category list
			Y_test_unsuffle.append(int(info[1]))

	# suffle test data
	index_shuf = range(len(X_test_unsuffle))
	shuffle(index_shuf)
	for i in index_shuf:
		X_test.append(X_test_unsuffle[i])
		Y_test.append(Y_test_unsuffle[i])

	# Convert class to categorical
	Y_train = np_utils.to_categorical(np.array(Y_train), nb_classes)
	Y_test = np_utils.to_categorical(np.array(Y_test), nb_classes)

	# Return data
	return np.array(X_train), Y_train, np.array(X_test), Y_test, nb_classes

# Export the KDEF in folds
def loadBatchKDEF(k, iteration):
	info = os.path.join(os.path.dirname(__file__), 'KDEF/info.txt')
	nb_classes = 7

	lines = None
	with open(info) as f:
		lines = f.read().splitlines()
		
	elByBatch = len(lines) / k
	start = elByBatch * iteration
	end = start + elByBatch
	
	test = lines[start:end]
	train = lines[0:start] + lines[end:len(lines)]

	X_train_unsuffle = []
	Y_train_unsuffle = []
	X_train = []
	Y_train = []

	# Open train info and split in lines
	for line in train:
		# split file info
		info = line.split('@')

		# Read the image
		img = Image.open(os.path.join(os.path.dirname(__file__), 'KDEF/'+info[0])).convert('L')

		# Append image to input list as a numpy array
		X_train_unsuffle.append(np.asarray(img).reshape(1, 100, 74))

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
	for line in test:
		# split file info
		info = line.split('@')

		# Read the image
		img = Image.open(os.path.join(os.path.dirname(__file__), 'KDEF/'+info[0])).convert('L')

		# Append image to input list as a numpy array
		X_test_unsuffle.append(np.asarray(img).reshape(1, 100, 74)) # FORMATO (CANAIS, ALTURA, LARGURA)

		# Append category to category list
		Y_test_unsuffle.append(int(info[1]))

	# suffle test data
	index_shuf = range(len(X_test_unsuffle))
	shuffle(index_shuf)
	for i in index_shuf:
		X_test.append(X_test_unsuffle[i])
		Y_test.append(Y_test_unsuffle[i])

	# Convert class to categorical
	Y_train = np_utils.to_categorical(np.array(Y_train), nb_classes)
	Y_test = np_utils.to_categorical(np.array(Y_test), nb_classes)

	# Return data
	return np.array(X_train), Y_train, np.array(X_test), Y_test, nb_classes
