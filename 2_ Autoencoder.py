# 导入numpay
import numpy as np
# 导入scikit-learn中的preprcessing模块，机器学习模块
import sklearn.preprocessing as prep
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 参数初始化,根据某一层网络的输入输出节点数量自动找到最合适的分布
# xavier是让权重满足０均值，方差为２/(n_in+n_out)
def xavier_init(fan_in, fan_out, constant=1):
    low = - constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    # 建立一个（low, high）的均匀分布，(max - min)^2/12正好满足方差
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)

# 定义自编码器的class
class AdditiveGaussianNoiseAutoencoder(object):
    # 输入变量数,隐含层节点数,隐含层激活函数,优化器,高斯噪声系数
    def __init__(self, n_input, n_hidden, transfer_function=tf.nn.softplus,
                 optimizer=tf.train.AdamOptimizer(), scale=0.1):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function
        self.scale = tf.placeholder(tf.float32)
        self.training_scale = scale
        network_weights = self._initialzie_weights()
        self.weights = network_weights
        # 定义网络
        # 为输入x创建一个维度为n_input的placeholder
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        # 建立能提取特征的隐含层,输入x加入噪声,然后与隐含层权重w1相乘,加上偏置b1
        self.hidden = self.transfer(tf.add(tf.matmul(self.x + scale * tf.random_normal((n_input,)),
                                                     self.weights['w1']), self.weights['b1']))
        # 在输出层对数据进行复原,重建操作,直接对隐藏层的输出self.hidden乘上输出层的权重w2,并加上偏置b2
        self.reconstruction = tf.add(tf.matmul(self.hidden,
                                               self.weights['w2']), self.weights['b2'])
        # 定义自编码器的损失函数,直接使用平方误差作为cost
        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0))
        # 定义训练操作,优化cost
        self.optimizer = optimizer.minimize(self.cost)
        # 创建session,初始化全部模型操作
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    #　参数初始化函数
    def _initialzie_weights(self):
        # 创建一个字典,将w1,w2,b1,b2存入,w1需用xavier进行初始化,其余全为0
        all_weights = dict()
        all_weights['w1'] = tf.Variable(xavier_init(self.n_input, self.n_hidden))
        all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden], dtype=tf.float32))
        all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden, self.n_input], dtype=tf.float32))
        all_weights['b2'] = tf.Variable(tf.zeros([self.n_input], dtype=tf.float32))
        return all_weights

    # 用一个batch执行一次训练并返回cost
    def partial_fit(self, X):
        cost, opt = self.sess.run((self.cost, self.optimizer),
                                  feed_dict={self.x: X, self.scale: self.training_scale})
        return cost

    # 只计算损失函数cost
    def calc_total_cost(self, X):
        return self.sess.run(self.cost, feed_dict={self.x: X, self.scale: self.training_scale})

    # 返回隐藏层的输出结果
    def transform(self, X):
        return self.sess.run(self.hidden, feed_dict={self.x: X, self.scale: self.training_scale})

    # 将隐藏层的输出作为输入,将高阶特征复原为原始数据
    def generate(self, hidden=None):
        if hidden is None:
            hidden = np.random.normal(size=self.weights["b1"])
        return self.sess.run(self.reconstruction, feed_dict={self.hidden: hidden})

    # 输出复原后的数据
    def reconstruct(self, X):
        return self.sess.run(self.reconstruction, feed_dict={self.x: X, self.scale: self.training_scale})

    #　获取w1
    def getweights(self):
        return self.sess.run(self.weights['w1'])

    # 获取b1
    def getBiases(self):
        return self.sess.run(self.weights['b1'])


# 读取数据集
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# 要保证噪声的一致性
# 对训练,测试数据进行标准化处理,标准化即将数据变成0均值,先减去均值,再除以标准差
def standard_scale(X_train, X_test):

    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    return X_train, X_test


#　定义一个随机整数,数据的随机性,不放回抽样
def get_random_block_from_data(data, batch_size):
    start_index = np.random.randint(0, len(data) - batch_size)
    return data[start_index: (start_index + batch_size)]

# 对数据集进行标准化处理
X_train, X_test = standard_scale(mnist.train.images, mnist.test.images)
# 定义常用参数
n_samples = int(mnist.train.num_examples)   # 总训练样本数
training_epochs = 100    # 最大训练轮数
batch_size = 128
display_step = 1

# 创建AGN自编码器的实例
autoencoder = AdditiveGaussianNoiseAutoencoder(n_input=784,     # 输入节点数
                                               n_hidden=100,    # 隐藏层节点数
                                               transfer_function=tf.nn.softplus,    #隐藏层激活函数
                                               optimizer=tf.train.AdamOptimizer(learning_rate=0.001),   # 优化器下降补偿
                                               scale=0.001)     # 噪声系数

# 开始训练
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(n_samples / batch_size)
    for i in range(total_batch):
        batch_xs = get_random_block_from_data(X_train, batch_size)

        cost = autoencoder.partial_fit(batch_xs)
        avg_cost += cost / n_samples * batch_size

    if epoch % display_step == 0:
        print("Epoch: ", '%04d' % (epoch + 1), "cost = ",
              "{:.9f}".format(avg_cost))

print("Total cost: " + str(autoencoder.calc_total_cost(X_test)))

