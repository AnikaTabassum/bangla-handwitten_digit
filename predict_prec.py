# make a prediction for a new image.
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from tensorflow.keras.models import *
import numpy as np


# load and prepare the image
def load_image(filename):
    # load the image
    img = load_img(filename, grayscale=True, target_size=(28, 28))
    # convert to array
    img = img_to_array(img)
    # print(type(img))
    # summarize shape
    # print(img.shape)
    # reshape into a single sample with 1 channel
    img = img.reshape(1, 28, 28, 1)
    # prepare pixel data
    img = img.astype('float32')
    img = img / 255.0
    return img


# load an image and predict the class
def run_example(i):
    # load the image
    # pp="D:\\8th_semester\\my_8th_semester\\Machine_Learning\\data\\Photos\\Photos\\"+str(i)+".jpg"
    # pp="D:\\8th_semester\\my_8th_semester\\Machine_Learning\\data\\digits\\"+str(i)+".jpg"
    pp = "D:\\8th_semester\\my_8th_semester\\Machine_Learning\\data\\0-9numbers\\0-9numbers\\" + str(
        i) + "_converted" + ".jpg"
    # pp="dataset (Bengali handwritten digit recognition)-20210410T171203Z-001\\2_two\\nipu_dgt_2__233.tif"
    # pp ="BDNet-master\\BDNet-master\\own\\own\\data\\nipu_dgt_5__85.jpg"
    img = load_image(pp)
    # load model
    model = load_model('models\\customModel7.model')
    # predict the class
    # digit = model.predict_classes(img)
    predictions = model.predict(img)
    # print(predictions)
    result = np.argmax(model.predict(img))
    print(str(i) + str(result))


# entry point, run the example
for i in range(0, 10):
    run_example(i)
