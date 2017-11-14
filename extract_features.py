import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

from keras.applications.vgg16 import VGG16, preprocess_input
from keras.applications.vgg19 import VGG19, preprocess_input
from keras.applications.xception import Xception, preprocess_input 
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.applications.mobilenet import MobileNet, preprocess_input
from keras.preprocessing import image
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.models import model_from_json
import _pickle   # adapted to python 3

from sklearn.preprocessing import LabelEncoder
import numpy as np
import glob
import cv2
import h5py
import os
import json
import datetime
import time
import gc

# [Optional] Handle GPU
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
gpu_config = tf.ConfigProto()
gpu_config.gpu_options.per_process_gpu_memory_fraction = 0.8
set_session(tf.Session(config=gpu_config))


# load the user configs
# with open('config/xception_config.json') as f:
# with open('config/vgg16_config.json') as f:
# with open('config/vgg19_config.json') as f:
# with open('config/inceptionv3_config.json') as f:
# with open('config/resnet50_config.json') as f:
# with open('config/inceptionresnetv2_config.json') as f:
with open('config/mobilenet_config.json') as f:
	config = json.load(f)

# config variables
model_name		= config["model"]
weights 		= config["weights"]
include_top 	= config["include_top"]
train_path 		= config["train_path"]
features_path	= config["features_path"]
labels_path 	= config["labels_path"]
test_size		= config["test_size"]
results			= config["results"]

# start time
print("[STATUS] start time - {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))) 
start = time.time()

# create the pretrained models
if model_name == "vgg16":
	base_model = VGG16(weights=weights)
	model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)
	image_size = (224, 224)
elif model_name == "vgg19":
	base_model = VGG19(weights=weights)
	model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)
	image_size = (224, 224)
elif model_name == "resnet50":
	base_model = ResNet50(weights=weights)
	model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)
	image_size = (224, 224)
elif model_name == "inceptionv3":
	base_model = InceptionV3(weights=weights)
	model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)
	image_size = (299, 299)
elif model_name == "xception":
	base_model = Xception(weights=weights)
	model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)
	image_size = (299, 299)
elif model_name == "inceptionresnetv2":
	base_model = InceptionResNetV2(weights=weights)
	model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)
	image_size = (299, 299)
elif model_name == "mobilenet":
	base_model = MobileNet(weights=weights)
	model = Model(inputs=base_model.input, outputs=base_model.get_layer('global_average_pooling2d_1').output)
	image_size = (224, 224)
else:
	base_model = None

# measure loading model time
load_model_time = time.time() - start
print("[INFO] loading model time: {}".format(load_model_time))
print("[INFO] successfully loaded base model: {}".format(model_name))

# path to training dataset
train_labels = os.listdir(train_path)

# encode the labels
print("[INFO] encoding labels...")
le = LabelEncoder()
le.fit([tl for tl in train_labels])

# variables to hold features and labels
features = []
labels   = []


# restart time counter
start = time.time()


# loop over all the labels in the folder
i = 0
for label in train_labels:
	cur_path = train_path + "/" + label
	for image_path in glob.glob(cur_path + "/*.jpg"):
		img = image.load_img(image_path, target_size=image_size)
		x = image.img_to_array(img)
		x = np.expand_dims(x, axis=0)
		x = preprocess_input(x)
		feature = model.predict(x)
		flat = feature.flatten()
		features.append(flat)
		labels.append(label)
		if i % 100 == 0:  # print every 100th processed image
			print("[INFO] processed - {}".format(i))
		i += 1
	print("[INFO] completed label - {}".format(label))

# encode the labels using LabelEncoder
targetNames = np.unique(labels)
le = LabelEncoder()
le_labels = le.fit_transform(labels)

# measure feature_extraction time
feature_extraction_time = time.time() - start
print("[INFO] Feature extraction time: {}".format(feature_extraction_time))


# get the shape of training labels
print ("[STATUS] training labels: {}".format(le_labels))
print ("[STATUS] training labels shape: {}".format(le_labels.shape))

# save features and labels 
h5f_data = h5py.File(features_path, 'w')
array_of_features = np.array(features)
h5f_data.create_dataset('dataset_1', data=array_of_features)

print("[INFO] Feature max value: {}".format(np.amax(array_of_features)))
print("[INFO] Feature min value: {}".format(np.amin(array_of_features)))

h5f_label = h5py.File(labels_path, 'w')
h5f_label.create_dataset('dataset_1', data=np.array(le_labels))

h5f_data.close()
h5f_label.close()

print ("[STATUS] features and labels saved..")

# end time
end = time.time()
print ("[STATUS] end time - {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))
print ("[INFO] Processing time \n")
print(" Loading model time: {}".format(load_model_time))
print(" Feature extraction time: {}".format(feature_extraction_time))
print(" Feature extraction time per sample: {}".format(feature_extraction_time/len(le_labels)))

# garbage collection
gc.collect()