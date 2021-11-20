
import pickle
import os.path
import numpy as np
import matplotlib.pyplot as plt
from imutils import paths
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers.convolutional import Conv2D, MaxPooling2D
from tensorflow.python.keras.layers.core import Flatten, Dense
import cv2
import pandas as pd
from mlxtend.data import loadlocal_mnist
from tensorflow.keras.models import load_model

X_train,Y_train=loadlocal_mnist(
          images_path='/content/drive/My Drive/MOSAIC_20/EMNIST/emnist-byclass-train-images-idx3-ubyte',
          labels_path='/content/drive/My Drive/MOSAIC_20/EMNIST/emnist-byclass-train-labels-idx1-ubyte'
)
X_test,Y_test=loadlocal_mnist(
          images_path='/content/drive/My Drive/MOSAIC_20/EMNIST/emnist-byclass-test-images-idx3-ubyte',
          labels_path='/content/drive/My Drive/MOSAIC_20/EMNIST/emnist-byclass-test-labels-idx1-ubyte'
)

X_train=X_train.reshape(len(X_train),28,28)
X_test=X_test.reshape(len(X_test),28,28)
for i in range(len(X_train)):
    X_train[i]=np.transpose(X_train[i])
for i in range(len(X_test)):
    X_test[i]=np.transpose(X_test[i])


X_train=X_train.reshape(len(X_train),28,28,1)/127.5-1.0
X_test=X_test.reshape(len(X_test),28,28,1)/127.5-1.0


label_category=np.array(['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','a','b','d','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z'])

model = Sequential()


model.add(Conv2D(40, (5, 5), padding="same", input_shape=(28,28,1), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2))) 


model.add(Conv2D(30, (5, 5), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(20, (5, 5), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Flatten())
model.add(Dense(500, activation="relu"))

model.add(Dense(len(label_category), activation="softmax"))



model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])


model.fit(X_train, Y_train, validation_split=0.35,shuffle=True,batch_size=32, epochs=10)

scores = model.evaluate(X_test, Y_test)
print("score ",scores)
