import matplotlib.pyplot as plt
import datetime
import os

'''
Save a Keras fit history as a CSV file
The columns is accuracy;loss;validation accuracy;validation loss
Automatic save a plot too
'''
def saveHistory(name, hist, model_number):
	print('Saving history')

	path = 'history/'+str(model_number)+'/histRAW'
	if not os.path.exists(path):
		os.makedirs(path)

	# Get the history values
	acc 	 = hist.history['acc']
	loss     = hist.history['loss']
	val_acc  = hist.history['val_acc']
	val_loss = hist.history['val_loss']

	# Save the CSV
	file = open(path+'/'+name+'.txt', 'w+')
	for i in range(0,len(acc)):
		file.write(str(acc[i]) + ';' + str(loss[i]) + ';' + str(val_acc[i]) + ';' + str(val_loss[i]) + '\n')
	file.close()

	# Save a plot from history
	plot(name, model_number, acc, val_acc, loss, val_loss)

'''
Plot a Keras fit history with MatPlotLib PyPlot
Use two differents plot, one to accuracy and other to loss
'''
def plot(name, model_number, acc, val_acc, loss, val_loss):
	print('Saving plots')

	path = 'history/'+str(model_number)
	if not os.path.exists(path):
		os.makedirs(path)

	# Accuracy plot
	plt.ylabel('Value')
	plt.xlabel('Epoch')
	plt.plot(acc, '-b', lw=2, label='Train Accuracy');
	plt.plot(val_acc, '-g', lw=2, label='Validation Accuracy');
	plt.legend(loc='lower right', borderaxespad=0.)
	plt.title('Accuracy with ' + name)

	plt.savefig(path+'/acc_'+name+'.png', dpi=300)
	plt.close()

	# Loss plot
	plt.ylabel('Value')
	plt.xlabel('Epoch')
	plt.plot(loss, '-b', lw=2, label='Train Loss');
	plt.plot(val_loss, '-g', lw=2, label='Validation Loss');
	plt.legend(loc='upper right', borderaxespad=0.)
	plt.title('Loss with ' + name)

	plt.savefig(path+'/loss_'+name+'.png', dpi=300)
	plt.close()

'''
Save the execute time
'''
def saveTime(info):
	file = open('history/time.txt', 'a+')
	file.write(info + '\n')
	file.close()
