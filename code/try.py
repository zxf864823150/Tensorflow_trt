import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt #导入可视化模块

def add_layer(inputs, in_size, out_size, activation_function=None):
	#activation_function=None表示没有激活函数  相当于是线性模型
	with tf.name_scope('layer'):
		with tf.name_scope('weights'):
			Weights = tf.Variable(tf.random_normal([in_size, out_size]), name = 'W')
		with tf.name_scope('biases'):
			biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name = 'b')#偏置最好不为零所以加了0.1
		with tf.name_scope('Wx_plus_b'):
			Wx_plus_b = tf.matmul(inputs, Weights) + biases
		if activation_function is None:
			outputs = Wx_plus_b
		else:
			outputs = activation_function(Wx_plus_b)
		return outputs

# 定义神经网络输入的placeholder
with tf.name_scope('inputs'):
	xs = tf.placeholder(tf.float32,[None,1], name = "x_input")
	#None表示可以输入任意的数据。因为x_data是300x1的矩阵，所以这里为[None,1]
	ys = tf.placeholder(tf.float32,[None,1], name = 'y_input')

#隐藏层layer1 输入节点1，输出节点10
l1 = add_layer(xs, 1, 10, activation_function = tf.nn.relu)
#预测的时候输入时隐藏层的输入l1,输入节点10，输出为y_data 有1个节点
prediction = add_layer(l1, 10, 1, activation_function = None)

#计算损失
with tf.name_scope('loss'):
	loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
						reduction_indices = [1]))
with tf.name_scope('train'):
	train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()
#tf.initialize_all_variables()运行的时候会提示你现在的新名称是tf.global_variables_initializer()
sess = tf.Session()
writer = tf.summary.FileWriter("./",sess.graph)
# 先加载到一个文件中，然后再加载到浏览器里观看，双引号里标出存放这个文件的路径
# 莫烦视频中是tf.train.SummaryWriter()，我运行报错后发现这个函数名称被改为了tf.summary.FileWriter()
sess.run(init)
###运行之后在命令行使用 tensorboard --logdir=/Users/zxf-pc/Desktop/tensorflow_shiyan/
###获得生成的图日志，并在浏览器中输入local:6006查看结构