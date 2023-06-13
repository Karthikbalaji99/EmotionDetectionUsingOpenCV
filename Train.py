import cv2
import os
import numpy as np
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense, BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras import regularizers
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

np.random.seed(42)

# Set the path to the data directory
d_path = 'Data'
categories = os.listdir(d_path)
labels = [i for i in range(len(categories))]

label_dict = dict(zip(categories, labels))

print(label_dict)
print(categories)
print(labels)

# Load and preprocess the images
data = []
target = []

for category in categories:
    f_path = os.path.join(d_path, category)
    img_names = os.listdir(f_path)
    for img_name in img_names:
        img_path = os.path.join(f_path, img_name)
        print("Loading image:", img_path)
        img = cv2.imread(img_path)

        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            data.append(img)
            target.append(label_dict[category])

        except Exception as e:
            print("Exception:", e)

data = np.array(data) / 255.0

target = np.array(target)
new_target = np_utils.to_categorical(target)

np.save('data', data)
np.save('newtarget', new_target)

# Training the model

data = np.load('data.npy')
target = np.load('newtarget.npy')

model = Sequential()

# Add more layers to the model

# ...
model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(48, 48,3)))
model.add(Conv2D(64,(3,3), padding='same', activation='relu' ))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128,(5,5), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
    
model.add(Conv2D(512,(3,3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(512,(3,3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten()) 
model.add(Dense(256,activation = 'relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))
    
model.add(Dense(512,activation = 'relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Dense(7, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Shuffle and split the data into training and testing sets
data, target = shuffle(data, target, random_state=42)
train_data, test_data, train_target, test_target = train_test_split(data, target, test_size=0.1, random_state=42)

# Define a callback to save the best model
checkpoint = ModelCheckpoint('Model-{epoch:2d}.model', monitor='val_loss', verbose=0, save_best_only=True, mode='auto')

# Train the model and save the best weights
history = model.fit(train_data, train_target, epochs=50, batch_size=16, callbacks=[checkpoint], validation_split=0.2)

# Plot the training history
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

# Evaluate the model on the test data
print(model.evaluate(test_data, test_target))
