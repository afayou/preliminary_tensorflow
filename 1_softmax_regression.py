from tensorflow.examples.tutorials.mnist import input_data  #导入数据集
import tensorflow as tf     # 载入tensorflow库
import time
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)  #数据集存放位置
print(mnist.train.images.shape, mnist.train.labels.shape)
print(mnist.test.images.shape, mnist.test.labels.shape)
print(mnist.validation.images.shape, mnist.validation.labels.shape)
# 创建新的session
sess = tf.InteractiveSession()
# 创建输入数据x的类型和维度
x = tf.placeholder(tf.float32, [None, 784])
# 创建w、b数据的类型和维度
w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
# 计算输出y，用softmax regession算法， w*x+b
y = tf.nn.softmax(tf.matmul(x, w) + b)
# 损失函数loss_function 实现cross-entropy
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
# 随机梯度下降算法SGD，学习速率理解为梯度下降的步长
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)
# 全局参数初始化
tf.global_variables_initializer().run()
# 执行训练，每次获取100个图像，执行1000次
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(1024)
    train_step.run({x: batch_xs, y_: batch_ys})
# 验证模型准确率
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))


A = [[0, 0, 1, 0, 1]]
B = [[1, 3, 4], [2, 4, 1]]

with tf.Session() as sess:
    print(sess.run(tf.argmax(A, 1)))
    print(sess.run(tf.argmax(B, 1)))
