import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import Sequential
from tensorflow.keras.losses import MeanSquaredError, SparseCategoricalCrossentropy
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import pandas as pd

# Importing train and test splits of the MNIST dataset and saving them as numpy arrays.
trainData = tfds.load('mnist', split='train')
testData = tfds.load('mnist', split='test')

# Converting the dataset into numpy arrays 
def convert_to_numpy(dataset):
    images, labels = [], []
    for example in tfds.as_numpy(dataset):
        images.append(example['image'].reshape(-1))  # Flatten
        labels.append(example['label'])
    return tf.constant(images), tf.constant(labels)

XTrain, yTrain = convert_to_numpy(trainData)
XTest, yTest = convert_to_numpy(testData)

# Building the NN layers with an l2 regularization
model = Sequential([
    Input(shape=(784,)),
    Dense(64, activation = 'relu', kernel_regularizer=l2(0.001)),
    Dense(32, activation = 'relu', kernel_regularizer=l2(0.001)),
    Dense(16, activation = 'relu', kernel_regularizer=l2(0.001)),
    Dense(10)
])

# Setting up the forward pass and training the model on the training set
model.compile(loss=SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
model.fit(XTrain, yTrain, epochs=100)


# Calculating the loss and accuracy of the model on the training set
training_loss, training_accuracy = model.evaluate(XTrain, yTrain)

# Calculating the loss and accuracy of the model on the test set
test_loss, test_accuracy = model.evaluate(XTest, yTest)
