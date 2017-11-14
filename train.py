# organize imports
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import numpy as np
import _pickle
import h5py
import os
import json
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import time


print("[STATUS] start time - {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))



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
test_size 	    = config["test_size"]
seed 		    = config["seed"]
features_path	= config["features_path"]
labels_path 	= config["labels_path"]
results  	    = config["results"]
classifier_path = config["classifier_path"]
train_path 	    = config["train_path"]
num_classes	    = config["num_classes"]

# import features and labels
h5f_data = h5py.File(features_path, 'r')
h5f_label = h5py.File(labels_path, 'r')

features_string = h5f_data['dataset_1']
labels_string   = h5f_label['dataset_1']

features = np.array(features_string)
labels   = np.array(labels_string)

h5f_data.close()
h5f_label.close()

# verify the shape of features and labels
print ("[INFO] Successfully loaded {} features.".format(model_name))
print ("[INFO] features shape: {}".format(features.shape))
print ("[INFO] labels shape: {}".format(labels.shape))

print ("[INFO] split into training and testing data...")
(trainData, testData, trainLabels, testLabels) = train_test_split(np.array(features),
                                                                  np.array(labels),
                                                                  test_size=test_size, 
                                                                  random_state=seed)

print ("[INFO] splitted train and test data...")
print ("[INFO] train data  : {}".format(trainData.shape))
print ("[INFO] test data   : {}".format(testData.shape))
print ("[INFO] train labels: {}".format(trainLabels.shape))
print ("[INFO] test labels : {}".format(testLabels.shape))


try:
    # load classifier from file, ie. logistic regression
    print("[INFO] loading classifier...")
    with open(classifier_path, 'rb') as fid:
        classifier_model = _pickle.load(fid)
except:
    print("[INFO] creating model/training...")
    classifier_model = LogisticRegression(random_state=seed)
    classifier_model.fit(trainData, trainLabels)

    # Save the model
    print("[INFO] dumping classifier...")
    f = open(classifier_path, "wb")
    f.write(_pickle.dumps(classifier_model))
    f.close()


# use rank-1 and rank-5 predictions
print("[INFO] evaluating model...")
f = open(results, "w")
rank_1 = 0


# loop over test data
start = time.time()
for (label, features) in zip(testLabels, testData):
	predictions = classifier_model.predict_proba(np.atleast_2d(features))[0]
	predictions = np.argsort(predictions)[::-1][:5]

	# rank-1 prediction increment
	if label == predictions[0]:
		rank_1 += 1


# convert accuracies to percentages
rank_1 = (rank_1 / float(len(testLabels))) * 100

# write the accuracies to file
f.write("rank-1: {}\n".format(rank_1))

# evaluate the model of test data
preds = classifier_model.predict(testData)

# write the classification report to file
print(classification_report(testLabels, preds))
f.write("{}\n".format(classification_report(testLabels, preds)))
f.close()

# measure prediction time
prediction_time = time.time() - start
print("[INFO] prediction time in sec: {}".format(prediction_time))
print("[INFO] prediction time per sample: {}".format(prediction_time/len(testLabels)))



# display the confusion matrix
print ("[INFO] confusion matrix")

# get the list of training lables
labels = sorted(list(os.listdir(train_path)))

# plot the confusion matrix
cm = confusion_matrix(testLabels, preds)
sns.heatmap(cm, 
            annot=True,
            cmap="Set2")
plt.show()
