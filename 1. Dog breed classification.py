# -*- coding: utf-8 -*-
"""
Created on Mon May  8 20:52:26 2023

@author: user
"""

#Import all relevant libraries
import numpy as np
import os
from itertools import chain
import random
import matplotlib.pyplot as plt
from matplotlib.image import imread
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical 
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import img_to_array, load_img
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras import models
from keras.optimizers import Adam
from keras.layers import GlobalAveragePooling2D, Dense, Dropout 
from keras.applications.inception_v3 import InceptionV3


#Load the stanford dog dataset
fpath = r"C:\Users\user\Downloads\archive (8)\images\Images"
print(os.listdir(fpath))
dog_classes = os.listdir(fpath)

#Separate to obtain the dog names to obtain the image path, their labels and the breed name
breeds = [breed.split('-',1)[1] for breed in dog_classes]
breeds

x = []
y = []

path = [r"C:\Users\user\Downloads\archive (8)\images\Images\{}".format(dog_class) for dog_class in dog_classes]

for counter, path in enumerate (path):
    for imgname in os.listdir(path):
        x.append([path + "\\" + imgname])
        y.append(breeds[counter])
        
print(x,"\n")
print(y,"\n")                 

#Checking the number of breeds (120) & images (20580)      
x = list(chain.from_iterable(x))
print(x,"\n")
len(x)

#Do random shuffle for machine learning 
combined = list (zip(x,y))
print(combined,"\n")
random.shuffle(combined)
print(combined,"\n")
x[:], y[:] = zip(*combined)

#Display dog pics randomly 
plt.figure(figsize=(18,18))

for counter, i in enumerate (random.sample(range(0, len(x)),9)):
    plt.subplot(3,3,counter+1)
    plt.subplots_adjust(hspace=0.3)
    filename = x[i]
    image = imread(filename)
    plt.imshow(image)
    plt.title(y[i], fontsize=12)

plt.show()


x = x[:1000]
y = y[:1000]

le = LabelEncoder()
le.fit(y)
y_ohe = to_categorical(le.transform(y), len(breeds))
print(y_ohe.shape)

y_ohe - np.array(y_ohe)

#Train, test, validate the models then check the shapes for each
img_data = np.array([img_to_array(load_img(img, target_size = (299,299))) for img in x])
x_train, x_test, y_train, y_test = train_test_split(img_data, y_ohe, test_size = 0.2, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

print(img_data.shape)
print(x_train.shape)
print(x_val.shape)
print(x_test.shape)
print(y_train.shape)
print(y_val.shape)
print(y_test.shape)

#Creating, training and validating generators 
batch_size = 32
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input, rotation_range=30, width_shift_range=0.2, height_shift_range=0.2,horizontal_flip= True)
train_generator = train_datagen.flow(x_train, y_train, shuffle = False, batch_size=batch_size, seed=1)

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input) 
val_generator = val_datagen.flow(x_val, y_val, shuffle= False, batch_size = batch_size, seed =1)

img_id=2 
dog_generator = train_datagen.flow(x_train[img_id:img_id+1], y_train[img_id:img_id+1], shuffle = False, batch_size = batch_size, seed=1)

plt.figure(figsize=(20,20))
dogs = [next(dog_generator) for i in range (0,5)]
for counter, dog in enumerate(dogs):
    plt.subplot (1,5, counter+1)
    plt.imshow(dog[0][0])
    
plt.show()

#Building the model with the pretrained data
base_model = InceptionV3(weights = 'imagenet', include_top = False, input_shape = (299,299,3))

model = models.Sequential()
model.add(base_model)
model.add(GlobalAveragePooling2D())
model.add(Dropout(0.3))
model.add(Dense(512, activation = 'relu'))
model.add(Dense(len(breeds), activation = 'softmax'))
model.layers[0].trainable = False

#Train Model 
model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
train_steps_per_epoch = x_train.shape[0] // batch_size
val_steps_per_epoch = len(x_val) // batch_size
epochs = 20 

history = model.fit_generator(train_generator, steps_per_epoch = train_steps_per_epoch, validation_data = val_generator, validation_steps = val_steps_per_epoch, epochs = epochs, verbose = 1)
