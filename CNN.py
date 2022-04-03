import tensorflow as tf
import numpy
from PIL import Image
import os

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

import time

directory = os.getcwd()
image_size = 128
kernal_size = 102

class_names = ['x', 'none_x']

# read in images from files
def readImage(folderName, action, imageArraySize):
	image_array = numpy.zeros(shape=(imageArraySize, image_size, image_size), dtype=int)
	counter = 0
	subdirectory = directory + '/images/' + folderName + '/' + action + '/'

	for filename in os.listdir(subdirectory):
		if filename.endswith(".png"):
			image = Image.open(subdirectory+filename).convert("L")
			np_image = numpy.array(image)
			image_array[counter, :, :] = np_image
		counter = counter + 1
	return image_array

def getData(folderList, no_image):
	train_images = numpy.zeros(shape=(1, image_size, image_size), dtype=int)
	test_images = numpy.zeros(shape=(1, image_size, image_size), dtype=int)

	for folder in folderList:
		action = "training"
		training_image = readImage(folder, action, no_image)
		train_images = numpy.concatenate((train_images, training_image), axis=0)
		action = "testing"
		testing_image = readImage(folder, action, no_image)
		test_images = numpy.concatenate((test_images, testing_image), axis=0)

	train_images = numpy.delete(train_images, 0, axis=0)
	test_images = numpy.delete(test_images, 0, axis=0)

	train_labels = numpy.array([])
	test_labels = numpy.array([])

	for i in range(0, len(folderList), 1):
		label = numpy.full((no_image,), i, dtype=int)
		train_labels = numpy.append(train_labels, label, axis=0)
		test_labels = numpy.append(test_labels, label, axis=0)

	train_labels = train_labels.astype(int)
	test_labels = test_labels.astype(int)

	train_images	= numpy.expand_dims(train_images.astype(numpy.float32) / 255.0, axis=3)
	test_images = numpy.expand_dims(test_images.astype(numpy.float32) / 255.0, axis=3)

	return train_images, train_labels, test_images, test_labels

def createCNNModel(train_images, train_labels, test_images, test_labels):

	# Include the epoch in the file name (uses `str.format`)
	checkpoint_path = str(image_size)+"_training/cp-{epoch:04d}.ckpt"
	checkpoint_dir = os.path.dirname(checkpoint_path)

		# Create a callback that saves the model's weights
	cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
																									 save_weights_only=True,
																									 verbose=1, period=1)

		# Create Convolutional Model
	model = models.Sequential()
	model.add(layers.Conv2D(128, (kernal_size, kernal_size), padding='valid' ,activation='relu', input_shape=(image_size, image_size, 1)))
	model.add(layers.MaxPooling2D((5, 5)))
	model.add(layers.Conv2D(64, (2, 2), activation='relu'))
	model.add(layers.MaxPooling2D((2, 2)))
	model.add(layers.Conv2D(64, (2, 2), activation='relu'))
	model.add(layers.Flatten())
	model.add(layers.Dense(64, activation='relu'))
	model.add(layers.Dense(2))

	model.summary()

		# Save the weights using the `checkpoint_path` format
	model.save_weights(checkpoint_path.format(epoch=0))

	# Start Timer
	start_time = time.time()

		# Compile and Train Model
	model.compile(optimizer='adam',
								loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
								metrics=['accuracy'])

	history = model.fit(train_images, train_labels, epochs=25, 
					validation_data=(test_images, test_labels), callbacks=[cp_callback])

	# Record Time after training model
	end_time = time.time() - start_time
	timeFile = open('Time_'+str(image_size)+'.txt', "w")
	timeFile.write("Recorded Training Time:	%.9f seconds" % end_time)
	timeFile.close()

	modelName = directory+'\\'+'model'
	if not os.path.exists(modelName):
		model.save(modelName)

	# Evaluate Model
	plt.figure()
	plt.plot(history.history['accuracy'], label='accuracy')
	plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
	plt.title("For Image Resolution "+ str(image_size) + "x" + str(image_size))
	plt.xlabel('Epoch')
	plt.ylabel('Accuracy')
	plt.ylim([0, 1.1])
	plt.xticks(numpy.arange(0, 25, step=1))
	plt.yticks(numpy.arange(0, 1.1, step=0.1))
	plt.legend(loc='lower right')
	plt.grid()
	plt.savefig(directory+'\\evaluate_model.png')
	plt.close()

	test_loss, test_acc = model.evaluate(test_images,	test_labels, verbose=2)

	print("test accuracy: ", test_acc)
	file = open(str(image_size)+"_accuracy.txt", 'w')
	for value in history.history['accuracy']:
		file.write(str(value)+"\n")
	file.close()

	return model

def plot_image(i, predictions_array, true_label, img, class_names):
	predictions_array, true_label, img = predictions_array, true_label[i], img[i]
	plt.grid(False)
	plt.xticks([])
	plt.yticks([])
	plt.imshow(img, cmap='gray')

	predicted_label = numpy.argmax(predictions_array)
	if predicted_label == true_label:
		color = 'blue'
	else:
		color = 'red'

	plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
																100*numpy.max(predictions_array),
																class_names[true_label]),
																color=color)

def plot_value_array(i, predictions_array, true_label, class_names):
	predictions_array, true_label = predictions_array, true_label[i]
	plt.grid(False)
	plt.xticks(range(len(class_names)), class_names)
	plt.yticks([])
	thisplot = plt.bar(range(len(class_names)), predictions_array, color="#777777")
	plt.ylim([0, 1])
	predicted_label = numpy.argmax(predictions_array)
	thisplot[predicted_label].set_color('red')
	thisplot[true_label].set_color('blue')

def trainCNN(folderList, no_image):
	# Create new Convolutional Neural Network and Trains CNN
	train_images, train_labels, test_images, test_labels = getData(folderList, no_image)
	model = createCNNModel(train_images, train_labels, test_images, test_labels)

	probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
	predictions = probability_model.predict(test_images)

	#train_images = numpy.squeeze(train_images, axis=3)
	test_images = numpy.squeeze(test_images, axis=3)
	#image = test_images[0]
	#plt.imshow(image, cmap='gray')
	#plt.show()

	i = 0
	#Verify predictions
	plt.figure(figsize=(6,3))
	plt.subplot(1,2,1)
	plot_image(i, predictions[i], test_labels, test_images, folderList)
	plt.subplot(1,2,2)
	plot_value_array(i, predictions[i],	test_labels, folderList)
	plt.show()

	num_rows = 6
	num_cols = 4
	num_images = num_rows*num_cols
	plt.figure(figsize=(2*2*num_cols, 2*num_rows))
	
	for i in range(num_images):
		plt.subplot(num_rows, 2*num_cols, 2*i+1)
		plot_image(i, predictions[i], test_labels, test_images, folderList)
		plt.subplot(num_rows, 2*num_cols, 2*i+2)
		plot_value_array(i, predictions[i], test_labels, folderList)
	plt.tight_layout()
	plt.show()

def createCNNModelOnly():
	model = models.Sequential()
	model.add(layers.Conv2D(128, (kernal_size, kernal_size), padding='valid' ,activation='relu', input_shape=(image_size, image_size, 1)))
	model.add(layers.MaxPooling2D((2, 2)))
	model.add(layers.Conv2D(64, (32, 32), activation='relu'))
	model.add(layers.MaxPooling2D((2, 2)))
	model.add(layers.Conv2D(64, (16, 16), activation='relu'))
	model.add(layers.Flatten())
	model.add(layers.Dense(64, activation='relu'))
	model.add(layers.Dense(2))

	model.summary()

	return model

def loadWeights(model, test_images, test_labels):
	file = open(str(image_size)+"_accuracy.txt", 'w')
	accuracy = numpy.array([])
	for epoch in range(0, 25, 1):
		checkpoint_path = str(image_size)+ "_training/cp-%04d.ckpt" % epoch
		model.load_weights(checkpoint_path)
		model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
							metrics=['accuracy'])
		loss, acc = model.evaluate(test_images, test_labels, verbose=2)
		print("Trained model, accuracy: {:5.2f}%".format(100*acc))
		accuracy = numpy.append(accuracy, acc)
		file.write(str(acc)+"\n")
	file.close()
	return accuracy

def loadWeight(model):
	value = 25
	checkpoint_path = str(image_size)+"_training/cp-%04d.ckpt" % value
	model.load_weights(checkpoint_path)
	model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
							metrics=['accuracy'])
	return model

def getImage(image_name):
	image_array = numpy.zeros(shape=(1, image_size, image_size), dtype=int)
	image = Image.open(image_name).convert("L")
	np_image = numpy.array(image)
	image_array[0, :, :] = np_image
	image_array = numpy.expand_dims(image_array.astype(numpy.float32) / 255.0, axis=3)

	return image_array

def plot_prediction(i, predictions_array, img):
	predictions_array, img = predictions_array, img[i]
	plt.grid(False)
	plt.xticks([])
	plt.yticks([])
	plt.imshow(img, cmap='gray')

def plot_array(i, predictions_array):
	predictions_array = predictions_array
	plt.grid(False)
	plt.xticks(range(len(class_names)), class_names)
	plt.yticks([])
	thisplot = plt.bar(range(len(class_names)), predictions_array, color="#777777")
	plt.ylim([0, 1])

def applyCNN(image_name):
	model = createCNNModelOnly()
	model = loadWeight(model)

	#accuracy_file = open("accuracy.txt", "w+")
	image = getImage(image_name)

	probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
	predictions = probability_model.predict(image)

	image = numpy.squeeze(image, axis=3)
	i = 0
	#Verify predictions
	plt.figure(figsize=(6,3))
	plt.subplot(1,2,1)
	plot_prediction(i, predictions[i], image)
	plt.subplot(1,2,2)
	plot_array(i, predictions[i])
	plt.show()


if __name__ == '__main__':
	no_image = image_size * 3
	trainCNN(class_names, no_image)
	#applyCNN('cropped_Figure_2.png_03_05.png')