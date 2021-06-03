#! /usr/bin/env python

# Import libraries
import keras
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from pandas import read_csv
from tensorflow.keras.optimizers import Adam
from sklearn.datasets import make_regression
from tensorflow.keras.models import Sequential
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Dropout, LSTM, Masking, Embedding

plt.rcParams.update({'font.size': 24})
plt.rcParams['font.family'] = ['serif']
plt.rcParams['font.serif'] = ['Times New Roman']

# Prepare and load data
dataframe = read_csv("trainingTP1OBS1solo.csv", delim_whitespace=False, header=None)
dataset = dataframe.values
print(dataset.shape)

# Split into input (X) and output (Y) variables
X = dataset[:, 0:94]
Y = dataset[:, 94:96]

# Graph linear and angular velocity training data
x_x = range(len(X))
plt.title("Linear and angular velocity training data")
plt.plot(x_x, Y[:,0], 'g', label="linear velocity")
plt.plot(x_x, Y[:,1], 'r', label="angular velocity")
plt.ylabel('linear velocity (m/s), angular velocity (rad/s)')
plt.xlabel('timesteps')
plt.legend()
plt.show()

# Convert input and output to numpy arrays of float type
XN = np.array(X, dtype = float)
YN = np.array(Y, dtype = float)

# Reshape the arrays into LSTM format (samples, timesteps, features)
XN = np.reshape(XN, (len(XN), 1, 94))
print("x:", XN.shape, "y:", YN.shape)

# Split training (80% data) and testing data (20% data)
x_train, x_test, y_train, y_test = train_test_split(XN, YN, test_size = 0.2, random_state = 4)

# Define LSTM sequential network model
# Network architecture: input layer with the 90 scan readings plus the target point and the mobile robot position in the free space
# Three hidden layers of 94 units each found by trial and error
# Dropout of 20% to reduce overfitting
model = Sequential()
model.add(LSTM(units = 188, return_sequences = True, input_shape = (1,94)))
model.add(Dropout(0.2))
model.add(LSTM(units = 94, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 94, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 94))
model.add(Dropout(0.2))
model.add(Dense(units = 2))
model.compile(loss = 'mean_squared_error', optimizer = 'adam', metrics = ['accuracy'])
model.summary()

# Fit model
history = model.fit(x_train, y_train, epochs = 500, validation_data = (x_test, y_test), batch_size = 16)

# Make predictions
ypred = model.predict(x_test)
print(ypred)

# Print mean squared error
print("y1 RMSE:%.4f".format(mean_squared_error(y_test[:,0], ypred[:,0], squared=False)))
print("y2 RMSE:%.4f".format(mean_squared_error(y_test[:,1], ypred[:,1], squared=False)))

# List all data in history
print(history.history.keys())

# Plot training loss vs test loss
plt.plot(history.history['loss'], linewidth=3.0)
plt.plot(history.history['val_loss'], linewidth=3.0)
plt.title('LSTM model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

# Save model and architecture to single file
model.save("trainingTP1OBS1solo2.h5")
print("Saved model to disk")
