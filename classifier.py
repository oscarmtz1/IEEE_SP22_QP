import os
import json
import numpy as np
import librosa
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras

def load_data(json_path):

    with open(json_path, 'r') as f:
        data = json.load(f)
    f.close()

    # Let's load our data into numpy arrays for TensorFlow compatibility.
    X = np.array(data["mfcc"])
    y = np.array(data["labels"])

    return X, y

model = tf.keras.models.load_model('saved_model/my_model')

music_file = ''
music_target = ''

inputs, targets = load_data(json_path=music_target)
stuff = train_test_split(inputs, targets, test_size=0.2)
model.predict(stuff[2])