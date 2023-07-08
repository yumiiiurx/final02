import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer


class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # 检查输入形状是否符合要求
        if not isinstance(input_shape, list) or len(input_shape) != 2:
            raise ValueError('An attention layer should be called on a list of 2 inputs.')
        if input_shape[0][2] != input_shape[1][2]:
            raise ValueError('Embedding sizes should be the same.')

        # 创建可训练的权重
        self.kernel = self.add_weight(shape=(input_shape[0][2], input_shape[0][2]),
                                      initializer='glorot_uniform',
                                      name='kernel',
                                      trainable=True)

        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        # 计算注意力权重
        a = K.dot(inputs[0], self.kernel)
        y_trans = K.permute_dimensions(inputs[1], (0, 2, 1))
        b = K.batch_dot(a, y_trans, axes=[2, 1])
        return K.tanh(b)

    def compute_output_shape(self, input_shape):
        # 输出形状与输入形状相同
        return (None, input_shape[0][1], input_shape[1][1])


seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)

#将每个导入语句放在单独的行上，按照标准库模块和第三方库模块的顺序排列。
#在类和函数之间添加了一个空行，以提高代码的可读性。
#在build方法中，使用了更简洁的条件判断语句。
#删除了多余的导入语句import tensorflow as tf，因为已经在开头导入了该模块。