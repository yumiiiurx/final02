from __future__ import print_function
import os
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.layers import Layer

seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)


class PositionWiseFeedForward(Layer):
    def __init__(self, model_dim, inner_dim, trainable=True, **kwargs):
        super(PositionWiseFeedForward, self).__init__(**kwargs)
        self._model_dim = model_dim
        self._inner_dim = inner_dim
        self._trainable = trainable

    def build(self, input_shape):
        self.weights_inner = self.add_weight(
            shape=(input_shape[-1], self._inner_dim),
            initializer='glorot_uniform',
            trainable=self._trainable,
            name="weights_inner"
        )
        self.weights_out = self.add_weight(
            shape=(self._inner_dim, self._model_dim),
            initializer='glorot_uniform',
            trainable=self._trainable,
            name="weights_out"
        )
        self.bais_inner = self.add_weight(
            shape=(self._inner_dim,),
            initializer='uniform',
            trainable=self._trainable,
            name="bais_inner"
        )
        self.bais_out = self.add_weight(
            shape=(self._model_dim,),
            initializer='uniform',
            trainable=self._trainable,
            name="bais_out"
        )
        super(PositionWiseFeedForward, self).build(input_shape)

    def call(self, inputs):
        if tf.keras.backend.dtype(inputs) != 'float32':
            inputs = tf.keras.backend.cast(inputs, 'float32')
        inner_out = tf.keras.backend.relu(tf.keras.backend.dot(inputs, self.weights_inner) + self.bais_inner)
        outputs = tf.keras.backend.dot(inner_out, self.weights_out) + self.bais_out
        print("==", outputs.shape)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape


'''
query = tf.random.truncated_normal([100, 50, 150])
w = PositionWiseFeedForward(150, 2048)(query)
print(w.shape)
'''
