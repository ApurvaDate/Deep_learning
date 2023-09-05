

#test whether GPU is working or not
"""

import tensorflow as tf

tf.test.gpu_device_name()

"""#which GPU we are using?"""

from tensorflow.python.client import device_lib

device_lib.list_local_devices()

"""#RAM information"""

!cat /proc/meminfo

!cat /proc/cpuinfo

from google.colab import drive
drive.mount('/content/drive')

!pip install -q keras

import pandas as pd

# part-1 : Data preprocessing

#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv("/content/drive/MyDrive/colab notebooks/ANN/Churn_Modelling.csv")
X = dataset.iloc[:,3:13]
y = dataset.iloc[:,13]

#create dummy variables
geography = pd.get_dummies(X['Geography'],drop_first=True)
gender = pd.get_dummies(X['Gender'],drop_first=True)

#concatenate the dataframes
X = pd.concat([X,geography,gender],axis=1)
X.head()

#drop unnecessary columns
X = X.drop(['Geography','Gender'],axis=1)
X.head()

#splitting the dataset into training and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=0)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

#Feature scaling
from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

#import keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
# from keras.layers import LeakyReLU, PReLU, ELU
from keras.layers import Dropout
#here tensorflow will be used as a backend
#tensorflow is an open source platform for machine learning, it can be used to create and deploy
# various types of machine learning models for different applications such as computer vision, nlp,
# recommender system and more.

#initialising the ANN
classifier = Sequential() #this is an empty neural network

#adding the first input layer and first hidden layer
# classifier.add(Dense(output_dim = 6, init = "he_uniform", activation = "relu", input_dim = 11))
#now output_dim has changed to units, init changed to kernel_initializer
classifier.add(Dense(units = 10, kernel_initializer = "he_normal", activation = "relu", input_dim = 11))
#here we are giving input_dim as 11 because we are using 11 features as an input
#adding a dropout layer
classifier.add(Dropout(0.3)) #while creating deep neural network with some ratio

#adding the second hidden layer
classifier.add(Dense(units = 20, kernel_initializer = "he_normal", activation = "relu"))
classifier.add(Dropout(0.4))

#adding the third hidden layer
classifier.add(Dense(units = 15, kernel_initializer = "he_normal", activation = "relu"))
classifier.add(Dropout(0.2))

#adding the ouput layer
classifier.add(Dense(units = 1, kernel_initializer = "glorot_uniform", activation = "sigmoid"))
#as this is a binary classification we need only one output neuron that is units=1

classifier.summary()

#compiling the ANN
classifier.compile(optimizer= "adam",loss= "binary_crossentropy", metrics= ['accuracy'])

# Commented out IPython magic to ensure Python compatibility.
# %%time
# #fitting the ANN to the training set
# # model_history = classifier.fit(X_train, y_train, validation_split = 0.33, batch_size= 10, nb_epoch = 100)
# model_history = classifier.fit(X_train, y_train, validation_split = 0.33, batch_size= 10, epochs = 100)

#Prediction on the test data
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

#making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

cm

#calculate the accuracy
from sklearn.metrics import accuracy_score
score = accuracy_score(y_pred, y_test)

score

