#!/usr/bin/env python3

import tensorflow as tf


class NMOutput(tf.keras.layers.Layer):
    def __init__(self, units=32):
        super().__init__()
        self.units = units

    def build(self, input_shape, nm_input_shape=(45,)):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True
        )
        self.b = self.add_weight(
            shape=(self.units,),
            initializer="zeros",
            trainable=True
        )
        self.ws = self.add_weight(
            shape=(nm_input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True
        )
        self.wb = self.add_weight(
            shape=(nm_input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True
        )

    def call(self, inputs, nm_inputs):
        return tf.matmul(inputs, self.w) \
            * (tf.matmul(nm_inputs, self.ws) + 1) + self.b \
            + tf.matmul(nm_inputs, self.wb)
