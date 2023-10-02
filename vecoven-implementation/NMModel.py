#!/usr/bin/env python3

import tensorflow as tf
from tensorflow.keras.layers import Dense, GRU

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

from Layers.NMOutput import NMOutput
from Layers.NMReLU import NMReLU


class NMModel(tf.keras.Model):
    def __init__(self, out_dim):
        super().__init__()
        self.gru_1 = GRU(100, return_sequences=True)
        self.gru_2 = GRU(75, return_sequences=True)
        self.relu = Dense(45, activation="relu")
        self.nmrelu_1 = NMReLU(30)
        self.nmrelu_2 = NMReLU(10)
        self.out = NMOutput(out_dim)

    def call(self, inputs):
        print("INPUT", inputs)
        if inputs[0,0].size > 2:
            z = self.gru_1(inputs)
            z = self.gru_2(z)
            z = self.relu(z)
        else:
            z = tf.zeros((1,1,45))
        y = self.nmrelu_1(inputs[:,:,-1], z)
        y = self.nmrelu_2(y, z)
        return self.out(y, z)
