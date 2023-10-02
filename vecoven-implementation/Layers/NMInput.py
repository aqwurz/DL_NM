#!/usr/bin/env python3

import tensorflow as tf


class NMInput(tf.keras.layers.Layer):
    def __init__(self, units=32):
        super().__init__()
        self.units = units

    def build(self, input_shape, nm_input_shape):
        self.b = self.add_weight(
            shape=(self.units,),
            initializer="zeros",
            trainable=True
        )
        self.ab = self.add_weight(
            shape=(self.units,),
            initializer="zeros",
            trainable=True
        )
        self.mb = self.add_weight(
            shape=(self.units,),
            initializer="zeros",
            trainable=True
        )
        self.amb = self.add_weight(
            shape=(self.units,),
            initializer="zeros",
            trainable=True
        )

    def call(self, inputs, prev_sb, prev_smb):
        return (inputs + self.b + tf.matmul(self.ab, prev_sb)) \
            * (self.mb + tf.matmul(self.ab, prev_smb))
