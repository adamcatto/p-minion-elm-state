#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 12:58:46 2020

@author: adamcatto
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, Dense
from tensorflow.keras.datasets import mnist


# LeNet-5 CNN architecture
def build_lenet_five():
    nn = Sequential()
    nn.add(Conv2D(filters=6, kernel_size=5, strides=1, activation='tanh',
                  input_shape=(28, 28, 1), padding='same'))
    nn.add(AveragePooling2D(pool_size=2, strides=2, padding='valid'))
    nn.add(Conv2D(filters=16, kernel_size=5, strides=1, activation='tanh',
                  input_shape=(14, 14, 6), padding='valid'))
    nn.add(AveragePooling2D(pool_size=2, strides=2, padding='valid'))
    nn.add(Conv2D(filters=120, kernel_size=5, strides=1, activation='tanh',
                  input_shape=(5, 5, 16), padding='valid'))
    nn.add(Flatten())
    nn.add(Dense(units=84, activation='tanh'))
    nn.add(Dense(units=10, activation='softmax'))
    return nn


build_lenet_five().summary()
