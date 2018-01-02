# in order to retrain the liveness detection network 
# please run this code 
# make sure you place all photos in to folders in
# liveness_detection folder 

import numpy as np 
import pandas as pd 
import os 
from tqdm import tqdm

import cv2

# I named the files with numbers
# becareful to put what you named


images = []
labels = []
for i in tqdm(range(1000,1100)):
    img = cv2.imread("liveness_detection/img-live/" + str(i)+".jpg", 0)
    
    img = cv2.resize(img, (100,100))
    images.append(img)
    labels.append(0)

for i in tqdm(range(1000,1100)):
    img = cv2.imread("liveness_detection/img-not-live/" + str(i)+".jpg", 0)
    img = cv2.resize(img, (100,100))
    images.append(img)
    labels.append(1)


X = np.array(images, dtype=float)

y = np.array(labels, dtype=float)


X /=255
y= y.reshape((-1,1))
X = X.reshape((-1,100,100,1))


from sklearn.preprocessing import OneHotEncoder
Oneencoder = OneHotEncoder()
y = Oneencoder.fit_transform(y)

print("Data is ready!")
print("Training is starting!")

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

# Building convolutional network
network = input_data(shape=[None, 100, 100, 1], name='input')
network = conv_2d(network, 32, 5, activation='relu')
network = avg_pool_2d(network, 2)
network = conv_2d(network, 64, 5, activation='relu')
network = avg_pool_2d(network, 2)
network = fully_connected(network, 128, activation='relu')
network = fully_connected(network, 64, activation='relu')
network = fully_connected(network, 2, activation='softmax',restore=False)
network = regression(network, optimizer='adam', learning_rate=0.0001,
                     loss='categorical_crossentropy', name='target')

model = tflearn.DNN(network, tensorboard_verbose=0)
model.load('model/my_model.tflearn')

model.fit(X, y.toarray(), n_epoch=3, validation_set=0.1, shuffle=True,
          show_metric=True, batch_size=32, snapshot_step=100,
          snapshot_epoch=False, run_id='model_finetuning')

model.save('model/my_model.tflearn')

print("Retraining is DONE!")
