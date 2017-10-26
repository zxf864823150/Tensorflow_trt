import tensorflow.examples.tutorials.mnist.input_data as input_data
import tensorflow as tf
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
x = tf.placeholder(tf.float32,[None,784])#placeholder表示占位符，tf的变量的传入需要站位，None表示任意维度
### 占位符可以理解为确定变量的输入形式，占位符获得位置，可以接受后部变量。
### Variable表示变量，并且初始化
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
### 使用softmax函数
y = tf.nn.softmax(tf.matmul(x,W) + b)
### 构架交叉熵损失函数
y_ = tf.placeholder("float", [None,10])
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
### 使用反向传播算法，以0.01位学习率。我觉得就是梯度下降法。
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
### 初始化所有变量
init = tf.global_variables_initializer()
# init = tf.initialize_all_variables()## 0.8版本形式
### 开始进行图计算
sess = tf.Session()
### 进行初始化
sess.run(init)
### 开始进行迭代，feed_dict是获得输入变量
for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(120)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
### 设置估计函数tf.argmax用于返回某个tensor对象的某一个维度上的最大值索引值。
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
###print(accuracy)这样输出的会是一个tensor对象，每一个tensor对象想要发挥必须要使用session，也就是图计算
print('精确如下')
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
print(mnist.test.labels.shape)
####
