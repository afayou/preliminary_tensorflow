from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# 导入数据集，创建默认的session
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
sess = tf.InteractiveSession()
# 初始化输入节点、隐藏层
in_units = 784
h1_units = 300
# 将w1初始化为截断的正态分布
w1 = tf.Variable(tf.truncated_normal([in_units, h1_units], stddev=0.1))
b1 = tf.Variable(tf.zeros([h1_units]))
w2 = tf.Variable(tf.zeros([h1_units, 10]))
b2 = tf.Variable(tf.zeros([10]))
# 定义输入x, 定义dropout的比率
x = tf.placeholder(tf.float32, [None, in_units])
keep_prob = tf.placeholder(tf.float32)
# 定义模型结构, 实现一个激活函数为relu的隐含层
hidden1 = tf.nn.relu(tf.matmul(x, w1) + b1)
# 调用dropout，随机将一部分节点置为0
hidden1_drop = tf.nn.dropout(hidden1, keep_prob)
# 定义输出层
y = tf.nn.softmax(tf.matmul(hidden1_drop, w2) + b2)
# 损失函数, cross_entropy 交叉损失熵
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
# 优化方法 自适应优化器
train_step = tf.train.AdagradOptimizer(0.3).minimize(cross_entropy)
# 全局初始化， 开始训练
tf.global_variables_initializer().run()
for i in range(3000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    train_step.run({x: batch_xs, y_: batch_ys, keep_prob: 0.75})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))



