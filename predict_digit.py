# make a prediction for a new image.
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from tensorflow.keras.models import *
import numpy as np
import os
# load and prepare the image
class predictDigit:
	def load_image(self,filename):
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
	def run_example(self, imageName):
		# load the image
		# pp="D:\\8th_semester\\my_8th_semester\\Machine_Learning\\data\\Photos\\Photos\\"+str(i)+".jpg"
		# pp="D:\\8th_semester\\my_8th_semester\\Machine_Learning\\data\\digits\\"+str(i)+".jpg"
		# pp="D:\\8th_semester\\my_8th_semester\\Machine_Learning\\data\\0-9numbers\\0-9numbers\\"+str(i)+"_converted"+".jpg"
		# pp="dataset (Bengali handwritten digit recognition)-20210410T171203Z-001\\2_two\\nipu_dgt_2__233.tif"
		# pp ="BDNet-master\\BDNet-master\\own\\own\\data\\nipu_dgt_5__85.jpg"
		img = self.load_image(imageName)
		# load model
		model = load_model(os.getcwd()+'\\models\\customModel8.model')
		# predict the class
		# digit = model.predict_classes(img)
		predictions = model.predict(img)
		# print(predictions)
		result= np.argmax(model.predict(img))
		print(str(result))
	 
	# entry point, run the example
	# for i in range(0,10):
i=9
imageName="D:\\8th_semester\\my_8th_semester\\Machine_Learning\\data\\0-9numbers\\0-9numbers\\"+str(i)+"_converted"+".jpg"
predictDigit().run_example(imageName)