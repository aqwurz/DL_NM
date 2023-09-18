#!/usr/bin/env python3

import tensorflow as tf

from .NMOutput import NMOutput


class NMReLU(NMOutput):
    def call(self, inputs, nm_inputs):
        return tf.nn.relu(super().call(inputs, nm_inputs))
