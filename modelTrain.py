import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
import os
class trainModel:
	def __init__(self, *args, **kwargs):
		print("initiating train")

		self.image_width=28 
		self.image_height= 28

		cwd = os.getcwd()
		print(cwd)
		self.train_data_directory= os.path.join(cwd, "dataset_train")
		print("t", self.train_data_directory)
		self.batch_size = 16
		self.initial_LR = 0.001
		self.epochs = 30

		self.dataGenerator = ImageDataGenerator(
			rescale=1 / 255.0,
			validation_split=0.3,
			rotation_range=20,
			zoom_range=0.15,
			width_shift_range=0.2,
			height_shift_range=0.2,
			shear_range=0.15,
			fill_mode="nearest"
		)

		self.trainGenerator = self.dataGenerator.flow_from_directory(
			self.train_data_directory,
			target_size=(self.image_width, self.image_height),
			color_mode="grayscale",
			batch_size=self.batch_size,
			subset="training",
			class_mode="categorical"
		)

		# # val_datagen = ImageDataGenerator(rescale=1. / 255)

		self.validationGenerator = self.dataGenerator.flow_from_directory(
			self.train_data_directory,
			target_size=(self.image_width, self.image_height),
			batch_size=self.batch_size,
			color_mode="grayscale",
			subset="validation",
			class_mode="categorical"
		)

	def plotTrainingLossAndAccuracy(self,modelHistory):
		print(modelHistory.history.keys())

		N = self.epochs
		plt.style.use("ggplot")
		plt.figure()
		plt.plot(np.arange(0, N), modelHistory.history["loss"], label="train_loss")
		plt.plot(np.arange(0, N), modelHistory.history["val_loss"], label="val_loss")
		plt.plot(np.arange(0, N), modelHistory.history["accuracy"], label="train_acc")
		plt.plot(np.arange(0, N), modelHistory.history["val_accuracy"], label="val_acc")
		plt.title("Training Loss and Accuracy")
		plt.xlabel("Epoch #")
		plt.ylabel("Loss/Accuracy")
		plt.legend(loc="lower left")
		plt.savefig(os.getcwd()+"\\plots\\plot8.png")
	# define cnn model

	def define_model(self):
		model = Sequential()
		model.add(Conv2D(32, (5, 5), strides=(1, 1),padding="same", activation='relu', input_shape=(28, 28, 1)))
		model.add(Conv2D(32, (5, 5), strides=(1, 1),padding="same", activation='relu',  input_shape=(28, 28, 1)))
		model.add(MaxPooling2D((5, 5)))
		model.add(Dropout(0.25))
		model.add(Conv2D(64, (5, 5), strides=(1, 1),padding="same", activation='relu', input_shape=(28, 28, 1)))
		model.add(Conv2D(64, (5, 5), strides=(1, 1),padding="same", activation='relu',  input_shape=(28, 28, 1)))
		model.add(MaxPooling2D((2, 2)))
		model.add(Dropout(0.25))
		model.add(Flatten())
		model.add(Dense(256, activation='relu'))
		model.add(Dropout(0.5))
		model.add(Dense(10, activation='softmax'))
		# compile model
		opt = Adam(learning_rate=0.001)
		model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
		return model

	def main(self):
		modelDirectory = os.getcwd()+"\\models"

		modelName = os.path.join(modelDirectory, 'customModel8.model')

		mainModel = self.define_model()
		# print(type(trainGenerator))
		modelHistory = mainModel.fit(self.trainGenerator,
									 steps_per_epoch=self.batch_size,
									 epochs=self.epochs, 
									 validation_data=self.validationGenerator)

		mainModel.save(modelName, save_format="h5")

		self.plotTrainingLossAndAccuracy(modelHistory)
		# print("Hello World!")

trainModel().main()