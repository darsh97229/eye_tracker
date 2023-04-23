import os
import cv2
import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
# Define the paths to the dataset and the output model
DATASET_PATH = '../Data/eyedb/mrlEyes_2018_01'
OUTPUT_MODEL_PATH = './eye_detection_model.h5'
# Define the size of the input images and the batch size for training
IMAGE_SIZE = (64, 64)
BATCH_SIZE = 32
# Define the CNN architecture for eye detection
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
# Compile the model with an appropriate loss function and optimizer
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
# Preprocess the dataset
def preprocess_dataset(dataset_path):
    print("Starting Preprocessing")
    images = []
    labels = []
    for folder_name in os.listdir(dataset_path):
        print("Processing folder:" + str(folder_name))
        folder_path = os.path.join(dataset_path, folder_name)
        for file_name in os.listdir(folder_path):
            #print("Processing file:" + str(file_name))
            file_path = os.path.join(folder_path, file_name)
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, IMAGE_SIZE)
            image = image / 255.0
            images.append(image)
            labels.append(int(file_name[16:17]))
            #print(int(file_name[16:17]))
    images = np.array(images).reshape((-1, IMAGE_SIZE[0], IMAGE_SIZE[1], 1))
    labels = np.array(labels)
    return images, labels
images, labels = preprocess_dataset(DATASET_PATH)
print("Finished Preprocessing")

# Split the dataset into training and validation sets
split_index = int(0.8 * len(images))
train_images, train_labels = images[:split_index], labels[:split_index]
val_images, val_labels = images[split_index:], labels[split_index:]
print("Finished splitting, now training")

# Train the model on the training set
model.fit(train_images, train_labels, epochs=10, batch_size=BATCH_SIZE,
          validation_data=(val_images, val_labels))
print("Finished training, now evaluating")

# Evaluate the model on the validation set
model.evaluate(val_images, val_labels)

print("Finished evaluating, now saving")
# Save the trained model to disk
model.save("eye_detection_model.h5")