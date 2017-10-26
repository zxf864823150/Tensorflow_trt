import tensorflow.examples.tutorials.mnist.input_data as input_data
import tensorflow as tf
### 加载数据，并且已经使用了one-hot热独编码表示
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
### 构建权值和偏离量
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
### 进行卷积和池化
### 卷积
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
### 池化
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')