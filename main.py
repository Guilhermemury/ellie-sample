# Convolutional Neural Network

#---------------- A. Train Model -----------------------------

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout
import os

# Constant Parameters
img_width, img_height = 150,150
batch_size=64

# 0. Initialising the CNN
classifier = Sequential()

# 1.

# convolutional layer - 1
classifier.add(Conv2D(32,(3,3),input_shape=(img_width, img_height,3),activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

# convolutional layer - 2
classifier.add(Conv2D(32,(3,3),activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

# convolutional layer - 3
classifier.add(Conv2D(64,(3,3),activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

# convolutional layer - 4
classifier.add(Conv2D(64,(3,3),activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

# 2. Flattening
classifier.add(Flatten())

# 3.Full connection
classifier.add(Dense(units = 64, activation = 'relu'))
classifier.add(Dropout(0.6))
classifier.add(Dense(units = 64, activation = 'relu'))
classifier.add(Dropout(0.3))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# 4. Compiling the CNN
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

# 5. Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
 rescale=1./255,
 shear_range=0.2,
 zoom_range=0.2,
 horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

# 6. Training & Test Sets

training_set = train_datagen.flow_from_directory(
 'images/train',
 target_size=(img_width, img_height),
 batch_size=batch_size,
 class_mode='binary')

test_set = test_datagen.flow_from_directory(
 'images/test',
 target_size=(img_width, img_height),
 batch_size=batch_size,
 class_mode='binary')

# Create a loss history
classifier.fit_generator(
 training_set,
 steps_per_epoch= 32000/batch_size,
 epochs=50,
 validation_data=test_set,
 validation_steps= 2000/batch_size)

#---------------- B. Save Model -----------------------------
model_save_path = os.path.join('jovem.h5')
classifier.save(model_save_path)
print("Model saved to", model_save_path)

#---------------- C. prediction -----------------------------
import numpy as np
from keras.preprocessing import image

test_image=image.load_img('1.jpg', target_size=(img_width, img_height))
test_image = image.img_to_array(test_image)

# Add fourth dimension
test_image = np.expand_dims(test_image,axis=0)
result= classifier.predict(test_image)
training_set.class_indices

# Result
result[0][0]
if result[0][0] ==1:
 prediction = 'jovem'
else:
 prediction = 'adulto'