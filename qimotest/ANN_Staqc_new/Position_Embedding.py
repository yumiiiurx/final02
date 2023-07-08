import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer

seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

class PositionEmbedding(Layer):
    def __init__(self, size=None, mode='sum', **kwargs):
        super(PositionEmbedding, self).__init__(**kwargs)
        self.size = size  # 必须为偶数
        self.mode = mode

    def call(self, x):
        if self.size is None or self.mode == 'sum':
            self.size = int(x.shape[-1])  # d_model的长度，比如512
        batch_size, seq_len = tf.shape(x)[0], tf.shape(x)[1]
        position_j = 1. / tf.pow(10000., 2 * tf.range(self.size / 2, dtype=tf.float32) / self.size)
        position_j = tf.expand_dims(position_j, 0)
        position_i = tf.cumsum(tf.ones_like(x[:, :, 0]), axis=1) - 1
        position_i = tf.expand_dims(position_i, 2)
        position_ij = tf.matmul(position_i, position_j)
        position_ij_2i = tf.sin(position_ij)[..., tf.newaxis]
        position_ij_2i_1 = tf.cos(position_ij)[..., tf.newaxis]
        position_ij = tf.concat([position_ij_2i, position_ij_2i_1], axis=-1)
        position_ij = tf.reshape(position_ij, (batch_size, seq_len, self.size))

        if self.mode == 'sum':
            return position_ij + x
        elif self.mode == 'concat':
            return tf.concat([position_ij, x], axis=-1)

    def compute_output_shape(self, input_shape):
        if self.mode == 'sum':
            return input_shape
        elif self.mode == 'concat':
            return (input_shape[0], input_shape[1], input_shape[2] + self.size)


'''
query = tf.random.truncated_normal([100, 50, 150])
w = PositionEmbedding(150, 'concat')(query)
print(w.shape)
'''
