#!/usr/bin/env python3

import tensorflow as tf
from tensorflow.keras.layers import Dense, GRU

from Layers.NMOutput import NMOutput
from Layers.NMReLU import NMReLU


class NMModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.gru_1 = GRU(100, return_sequences=True)
        self.gru_2 = GRU(75, return_sequences=True)
        self.relu = Dense(45, activation="relu")
        self.nmrelu_1 = NMReLU(30)
        self.nmrelu_2 = NMReLU(10)
        self.out = NMOutput(1)

    def call(self, inputs):
        z = self.gru_1(inputs)
        z = self.gru_2(z)
        z = self.relu(z)
        y = self.nmrelu_1(inputs, z)
        y = self.nmrelu_2(y, z)
        return self.out(y, z)
