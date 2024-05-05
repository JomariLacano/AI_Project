import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import to_categorical

# path to images
path = 'datasets/animals/'

# animal categories
categories = ['cats', 'dogs', 'panda']

# Display some pictures
for category in categories:
    fig, _ = plt.subplots(3,4)
    fig.suptitle(category)
    fig.patch.set_facecolor('xkcd:white')
    for k, v in enumerate(os.listdir(path+category)[:12]):
        img = plt.imread(path+category+'/'+v)
        plt.subplot(3, 4, k+1)
        plt.axis('off')
        plt.imshow(img)
    plt.show()
    
    shape0 = []
    shape1 = []

for category in categories:
    for files in os.listdir(path+category):
        shape0.append(plt.imread(path+category+'/'+ files).shape[0])
        shape1.append(plt.imread(path+category+'/'+ files).shape[1])
    print(category, ' => height min : ', min(shape0), 'width min : ', min(shape1))
    print(category, ' => height max : ', max(shape0), 'width max : ', max(shape1))
    shape0 = []
    shape1 = []
    
# initialize the data and labels
data = []
labels = []
imagePaths = []
HEIGHT = 32
WIDTH = 55
N_CHANNELS = 3

# grab the image paths and randomly shuffle them
for k, category in enumerate(categories):
    for f in os.listdir(path+category):
        imagePaths.append([path+category+'/'+f, k]) # k=0 : 'dogs', k=1 : 'panda', k=2 : 'cats'

import random
random.shuffle(imagePaths)
print(imagePaths[:10])

# loop over the input images
for imagePath in imagePaths:
    # load the image, resize the image to be HEIGHT * WIDTH pixels (ignoring
    # aspect ratio) and store the image in the data list
    image = cv2.imread(imagePath[0])
    image = cv2.resize(image, (WIDTH, HEIGHT))  # .flatten()
    data.append(image)
    
    # extract the class label from the image path and update the
    # labels list
    label = imagePath[1]
    labels.append(label)

# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# Let's check everything is ok
fig, _ = plt.subplots(3,4)
fig.suptitle("Sample Input")
fig.patch.set_facecolor('xkcd:white')
for i in range(12):
    plt.subplot(3,4, i+1)
    plt.imshow(data[i])
    plt.axis('off')
    plt.title(categories[labels[i]])
plt.show()

# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, random_state=42)

# Preprocess class labels
trainY = to_categorical(trainY, num_classes=3)

print(trainX.shape)
print(testX.shape)
print(trainY.shape)
print(testY.shape)

model = Sequential()
model.add(Convolution2D(32, (2, 2), activation='relu', input_shape=(HEIGHT, WIDTH, N_CHANNELS)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(32, (2, 2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())
model.fit(trainX, trainY, batch_size=32, epochs=25, verbose=1)

# Train the model
history = model.fit(trainX, trainY, batch_size=32, epochs=25, verbose=1, validation_data=(testX, to_categorical(testY, num_classes=3)))

# Save the model
model.save("animal_classification_model.h5")

# Plot training accuracy and loss
plt.plot(history.history['accuracy'])
plt.plot(history.history['loss'])
plt.title('Model Accuracy and Loss')
plt.xlabel('Epoch')
plt.legend(['Accuracy', 'Loss'], loc='upper left')
plt.savefig('animals/training_accuracy_loss.png')
plt.show()

# Plot test accuracy and loss
plt.plot(history.history['val_accuracy'])
plt.plot(history.history['val_loss'])
plt.title('Model Validation Accuracy and Loss')
plt.xlabel('Epoch')
plt.legend(['Val Accuracy', 'Val Loss'], loc='upper left')
plt.savefig('animals/validation_accuracy_loss.png')
plt.show()

# Sample predictions on the test set
predictions = model.predict(testX)
fig, axes = plt.subplots(3, 4)
fig.suptitle("Sample Predictions on Test Set")
fig.patch.set_facecolor('xkcd:white')
for i, ax in enumerate(axes.flat):
    ax.imshow(testX[i])
    ax.axis('off')
    true_label = categories[testY[i]]
    predicted_label = categories[np.argmax(predictions[i])]
    ax.set_title(f'True: {true_label}\nPredicted: {predicted_label}')
plt.savefig('animals/sample_predictions.png')
plt.show()
