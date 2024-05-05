import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.optimizers import SGD
import os

# Load dataset
(trainX, trainY), (testX, testY) = cifar10.load_data()
trainY = to_categorical(trainY)
testY = to_categorical(testY)

# Scale pixels
trainX = trainX.astype('float32') / 255.0
testX = testX.astype('float32') / 255.0

# Define model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(10, activation='softmax'))
opt = SGD(learning_rate=0.001, momentum=0.9)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# Fit model
history = model.fit(trainX, trainY, epochs=25, batch_size=64, verbose=1)

# Plot training loss
plt.plot(history.history['loss'], label='train')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
if not os.path.exists('cifar10'):
    os.makedirs('cifar10')
plt.savefig('cifar10/training_loss.png')
plt.close()

# Plot training accuracy
plt.plot(history.history['accuracy'], label='train')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('cifar10/training_accuracy.png')
plt.close()

# Choose random samples for prediction
sample_indices = np.random.choice(len(testX), 5)
sample_images = testX[sample_indices]
sample_labels = testY[sample_indices]

# Make predictions
predictions = model.predict(sample_images)

# Plot and save sample predictions
for i in range(len(sample_indices)):
    plt.imshow(sample_images[i])
    true_label = np.argmax(sample_labels[i])
    pred_label = np.argmax(predictions[i])
    plt.title(f'True: {true_label}, Predicted: {pred_label}')
    plt.savefig(f'cifar10/cifar10_sample_prediction_{i}.png')
    plt.close()

# Save model
model.save('cifar10/cifar10_model_new.h5')
 