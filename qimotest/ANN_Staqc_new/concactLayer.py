import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer


class ConcatLayer(Layer):   # 修改类名，遵循驼峰命名法
    def __init__(self, **kwargs):
        super(ConcatLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ConcatLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        # 按照指定的轴将输入张量切割成多个张量
        block_level_code_output = tf.split(inputs, inputs.shape[1], axis=1)
        # 按照指定的轴将张量拼接起来
        block_level_code_output = tf.concat(block_level_code_output, axis=2)
        # 去除维度为1的维度
        block_level_code_output = tf.squeeze(block_level_code_output, axis=1)
        print(block_level_code_output)
        return block_level_code_output

    def compute_output_shape(self, input_shape):
        print("===========================", input_shape)
        return (input_shape[0], input_shape[1] * input_shape[2])


seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)

#将from keras.layers import *修改为from tensorflow.keras.layers import Layer，以使用完整的模块导入并指定Layer类的来源。
#将from keras import backend as K修改为import tensorflow as tf，以使用 TensorFlow 的后端函数。
#调整了导入模块的顺序，按照标准库模块和第三方库模块的顺序排列。
#将class concatLayer(Layer)修改为class ConcatLayer(Layer)，遵循类名的驼峰命名法。
#修改了 compute_output_shape 方法的返回语句，使其返回一个元组。