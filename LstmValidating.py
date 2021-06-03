#! /usr/bin/env python

# Import libraries
import csv
import keras
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from pandas import read_csv
from keras.utils import plot_model
from tensorflow.keras.models import load_model

plt.rcParams.update({'font.size': 26})
plt.rcParams['font.family'] = ['serif']
plt.rcParams['font.serif'] = ['Times New Roman']

# Load neural network model
model = load_model('trainingTP1OBS1solo.h5')

# Summarize model
model.summary()
# tf.keras.utils.plot_model(model, to_file='trainingTP1-2-9-10-14-15OBS1-2-3.png', show_shapes=True, show_layer_names=True)

# Prepare and load data
dataframe = read_csv("trainingTP1OBS1solo.csv", delim_whitespace=False, header=None)
dataset = dataframe.values

# Split into input (X) and output (Y) variables
X = dataset[:, 0:94]
Y = dataset[:, 94:96]

# Graph linear and angular velocity training data
x_x = range(len(X))
plt.title("Linear and angular velocity training data")
plt.plot(x_x, Y[:,0], 'g', label="human-driver linear velocity", linewidth=1.0)
plt.plot(x_x, Y[:,1], 'r', label="human-driver angular velocity", linewidth=1.0)
plt.ylabel('linear velocity (m/s), angular velocity (rad/s)')
plt.xlabel('timesteps')
plt.legend()
plt.show()

def Average(CACA):
    return sum(CACA) / len(CACA)

# Convert input and output to numpy arrays of float type
XN = np.array(X, dtype = float)
YN = np.array(Y, dtype = float)

# Reshape the arrays into LSTM format (samples, timesteps, features)
XN = np.reshape(XN, (len(XN), 1, 94))
print("x:", XN.shape, "y:", YN.shape)

# Evaluate the model
score = model.evaluate(XN, YN, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

# Make predictions
ypred = model.predict(XN)
# print(ypred)

print(Average(Y))

# Graph LSTM multi-output prediction
x_ax = range(len(XN))
plt.title("LSTM multi-output prediction")
plt.plot(x_ax, YN[:,0], 'g', label="human-driver linear velocity", linewidth=1.0)
plt.plot(x_ax, ypred[:,0], 'r', label="LSTM predicted linear velocity", linewidth=1.0, linestyle='--')
plt.plot(x_ax, YN[:,1], 'b', label="human-driver angular velocity", linewidth=1.0)
plt.plot(x_ax, ypred[:,1], 'm', label="LSTM predicted angular velocity", linewidth=1.0, linestyle='--')
plt.ylabel('linear velocity (m/s), angular velocity (rad/s)')
plt.xlabel('timesteps')
plt.legend()
plt.show()
