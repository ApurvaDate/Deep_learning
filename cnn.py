# -*- coding: utf-8 -*-
"""CNN.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ZHtH7BputVzJHFsRHRRskeOm8pBP_nA6
"""

!pip install keras-tuner

import tensorflow as tf

from tensorflow import keras
import numpy as np

print(tf.__version__)

fashion_mnist = keras.datasets.fashion_mnist

(train_images,train_labels),(test_images,test_labels) = fashion_mnist.load_data()

train_images = train_images/255.0
test_images = test_images/255.0

train_images[0]

train_images.shape #this is 28 , 28 pixels

#reshape the images
train_images = train_images.reshape(len(train_images),28,28,1)
test_images = test_images.reshape(len(test_images),28,28,1)

#create a function called build model
#hp is hyperparameter
def build_model(hp):
  model = keras.Sequential([
      keras.layers.Conv2D(
          filters=hp.Int('conv_1_filter',min_value=32,max_value=128, step=16),
          kernel_size=hp.Choice('conv_1_kernel',values = [3,5]),
          activation = 'relu',
          input_shape=(28,28,1)
        ),
      keras.layers.Conv2D(
          filters = hp.Int('conv_2_filter',min_value=32, max_value =64,step=16),
          kernel_size=hp.Choice('conv_2_kernel', values = [3,5]),
          activation = 'relu'
      ),
      keras.layers.Flatten(),
      keras.layers.Dense(
          units=hp.Int('dense_1_units',min_value=32,max_value=128,step=16),
          activation='relu'
      ),
      keras.layers.Dense(10,activation='softmax')
  ])
  model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate',values=[1e-2,1e-3])),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
  return model

from kerastuner import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters

tuner_search = RandomSearch(build_model,objective = 'val_accuracy',max_trials=5,directory='output', project_name = 'Mnist Fashion')

tuner_search.search(train_images,train_labels,epochs=3,validation_split=0.1) #the object will try to search best parameter

model=tuner_search.get_best_models(num_models=1)[0]

model.summary()  #here we get the best model, and after that we retrain the model with these best parameters

model.fit(train_images,train_labels,epochs=10,validation_split=0.1,initial_epoch=3)

