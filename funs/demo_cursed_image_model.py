from keras.preprocessing.image import img_to_array,load_img
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import json
import os
import keras
from keras.utils.generic_utils import CustomObjectScope
from keras.models import load_model

def load_image(image_path):
	'''
	Loads and preprocesses an image to run the model on
	Input: Path to the image file to rn the model on
	Output: a copy of the original and the preprocessed image to use for
	model prediction
	'''
	try:
		image = cv2.imread(image_path)
		orig  = image.copy()
		image = cv2.resize(image,(224,224))
		image = image.astype("float")/255.0
		image = img_to_array(image)
		image = np.expand_dims(image,axis=0)
		return image,orig
	except:
		raise

def load_pretrained_model(model_path):
	'''
	Input: path to the json file and the hdf5 file for the trained semantic salience map
	Output: a trained model
	'''
	try:
		path_to_json    = os.path.expanduser(model_path)+'cursed_image_model.json'
		path_to_weights = os.path.expanduser(model_path)+'cursed_image_model.h5'
		with CustomObjectScope({'relu6': keras.applications.mobilenet.mobilenet.relu6,'DepthwiseConv2D': keras.layers.DepthwiseConv2D}):
			model = load_model(path_to_weights)
		return model
	except:
		raise

def generate_class_prediction(image,model):
	'''
	Function to generate a class preciction and confidence for an image
	Input: the preprocessed input image array and the trained model
	Output: the predicted class and class confidence values
	'''
	try:
		preds_conf  = model.predict(image)[0][0]
		preds_class = model.predict_classes(image)
		#Setup the string label
		if preds_class == 0:
			text_class  = 'Cursed'
			probability = (1 - preds_conf)*100
		else:
			text_class  = 'Neutral'
			probability = preds_conf*100
		text_label = "{}: {:.2f}%".format(text_class, probability)
		return text_label
	except:
		raise

def display_image_with_probability(original_image,text_label):
	'''
	Draws image to the screen with the predicted class label
	Input: the original image and the text label with the predicted class name and probability
	'''
	try:
		cv2.putText(original_image,text_label,(10,25),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
		cv2.imshow("Output",original_image)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
	except:
		raise

def main():
	'''
	Load the pretrained model, predict the class, and display the original image with the 
	predicted class to the screen
	'''
	try:
		#Argument parsing
		ap = argparse.ArgumentParser()
		ap.add_argument("-m", "--model_path", default='out/',
			help="path to trained model model")
		ap.add_argument("-i", "--image_path", required=True,
			help="path to input image")
		args = vars(ap.parse_args())

		#Load and preprocess the image for prediction
		image,orig = load_image(args['image_path'])
		model      = load_pretrained_model(args['model_path'])
		text_label = generate_class_prediction(image,model)
		# display_image_with_probability(orig,model)
		display_image_with_probability(orig,text_label)
	except:
		raise

if __name__ == '__main__':
	main()