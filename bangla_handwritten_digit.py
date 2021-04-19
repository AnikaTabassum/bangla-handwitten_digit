import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
import os

image_width, image_height = 28, 28
train_data_directory = 'D:\\8th_semester\\my_8th_semester\\Machine_Learning\\dataset (Bengali handwritten digit recognition)-20210410T171203Z-001'

batch_size = 16
initial_LR = 0.001
epochs = 25

dataGenerator = ImageDataGenerator(
    rescale=1 / 255.0,
    validation_split=0.3
)

trainGenerator = dataGenerator.flow_from_directory(
    train_data_directory,
    target_size=(image_width, image_height),
    color_mode="grayscale",
    batch_size=batch_size,
    subset="training",
    class_mode="categorical"
)

# # val_datagen = ImageDataGenerator(rescale=1. / 255)

validationGenerator = dataGenerator.flow_from_directory(
    train_data_directory,
    target_size=(image_width, image_height),
    batch_size=batch_size,
    color_mode="grayscale",
    subset="validation",
    class_mode="categorical"
)


# define cnn model
def define_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))
    # compile model
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


modelDirectory = 'D:\\8th_semester\\my_8th_semester\\Machine_Learning\\dataset (Bengali handwritten digit recognition)-20210410T171203Z-001'

modelName = os.path.join(modelDirectory, 'models/customModel.model')

mainModel = define_model()
print(type(trainGenerator))
modelHistory = mainModel.fit(trainGenerator,
                             steps_per_epoch=batch_size,
                             epochs=epochs,
                             validation_data=validationGenerator)

mainModel.save(modelName, save_format="h5")
# plotTrainingLossAndAccuracy(modelHistory)
