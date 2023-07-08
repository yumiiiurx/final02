import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer

class MediumLayer(Layer):
    def __init__(self, **kwargs):
        super(MediumLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(MediumLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        sentence_token_level_outputs = tf.stack(inputs, axis=0)
        sentence_token_level_outputs = K.permute_dimensions(sentence_token_level_outputs, (1, 0, 2))
        return sentence_token_level_outputs

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], len(input_shape), input_shape[0][1])

'''
x1 = tf.random.truncated_normal([100, 150])
x2 = tf.random.truncated_normal([100, 150])
x3 = tf.random.truncated_normal([100, 150])
x4 = tf.random.truncated_normal([100, 150])

w = MediumLayer()([x1, x2, x3, x4])
print(w)
'''


#引入语句from tensorflow.keras.layers import *改为明确导入from tensorflow.keras.layers import Layer。这样可以避免导入不必要的符号，提高代码的可读性。
#将import tensorflow as tf语句移到from tensorflow.keras.layers import Layer之前。按照惯例，标准库导入应该放在第一行。
#修正了代码中的缩进，确保代码块的缩进为4个空格。

