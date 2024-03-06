# srcnn.py

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D


def build_srcnn(input_shape=(None, None, 1)):
    model = Sequential([
        Conv2D(filters=128, kernel_size=(9, 9), activation='relu', padding='same', input_shape=input_shape),
        Conv2D(filters=64, kernel_size=(1, 1), activation='relu', padding='same'),
        Conv2D(filters=1, kernel_size=(5, 5), activation='linear', padding='same')
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model
