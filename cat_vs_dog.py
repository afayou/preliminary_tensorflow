import tensorflow as tf
import math
import numpy as np
import time
import os
# 定义训练轮数，每轮处理的数据量，数据集下载位置
max_step = 10000
batch_size = 8
N_CLASSES = 2
IMG_SIZE = 224
CAPACITY = 256
image_dir = 'data/train/'
test_dir = 'data/test/'

def get_all_files(file_path, is_random=True):
    """
    获取图片路径及其标签
    :param file_path: a sting, 图片所在目录
    :param is_random: True or False, 是否乱序
    :return:
    """
    image_list = []
    label_list = []

    cat_count = 0
    dog_count = 0
    for item in os.listdir(file_path):
        item_path = file_path + '/' + item
        item_label = item.split('.')[0]  # 文件名形如  cat.0.jpg,只需要取第一个

        if os.path.isfile(item_path):
            image_list.append(item_path)
        else:
            raise ValueError('文件夹中有非文件项.')

        if item_label == 'cat':  # 猫标记为'0'
            label_list.append(0)
            cat_count += 1
        else:  # 狗标记为'1'
            label_list.append(1)
            dog_count += 1
    print('数据集中有%d只猫,%d只狗.' % (cat_count, dog_count))

    image_list = np.asarray(image_list)
    label_list = np.asarray(label_list)
    # 乱序文件
    if is_random:
        rnd_index = np.arange(len(image_list))
        np.random.shuffle(rnd_index)
        image_list = image_list[rnd_index]
        label_list = label_list[rnd_index]

    return image_list, label_list


def get_batch(train_list, image_size, batch_size, capacity, is_random=True):
    """
    获取训练批次
    :param train_list: 2-D list, [image_list, label_list]
    :param image_size: a int, 训练图像大小
    :param batch_size: a int, 每个批次包含的样本数量
    :param capacity: a int, 队列容量
    :param is_random: True or False, 是否乱序
    :return:
    """

    intput_queue = tf.train.slice_input_producer(train_list, shuffle=False)

    # 从路径中读取图片
    image_train = tf.read_file(intput_queue[0])
    image_train = tf.image.decode_jpeg(image_train, channels=3)  # 这里是jpg格式
    image_train = tf.image.resize_images(image_train, [image_size, image_size])
    image_train = tf.cast(image_train, tf.float32) / 255.  # 转换数据类型并归一化

    # 图片标签
    label_train = intput_queue[1]

    # 获取批次
    if is_random:
        image_train_batch, label_train_batch = tf.train.shuffle_batch([image_train, label_train],
                                                                      batch_size=batch_size,
                                                                      capacity=capacity,
                                                                      min_after_dequeue=100,
                                                                      num_threads=2)
    else:
        image_train_batch, label_train_batch = tf.train.batch([image_train, label_train],
                                                              batch_size=1,
                                                              capacity=capacity,
                                                              num_threads=1)
    return image_train_batch, label_train_batch

# 定义初始化weight的函数，使用L2正则处理
def variable_with_weight_loss(shape, stddev, w1):
    var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    if w1 is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(var), w1, name='weight_loss')
        tf.add_to_collection('losses', weight_loss)
    return var

train_list = get_all_files(image_dir, True)
test_list = get_all_files(image_dir, True)
images_train, labels_train = get_batch(train_list, IMG_SIZE, batch_size, CAPACITY, True)
images_test, labels_test = get_batch(test_list, IMG_SIZE, batch_size, CAPACITY, True)
# 创建输入数据的placeholder
image_holder = tf.placeholder(tf.float32, [batch_size, IMG_SIZE, IMG_SIZE, 3])
label_holder = tf.placeholder(tf.int32, [batch_size])
# 创建第一个卷积层
weight1 = variable_with_weight_loss(shape=[5, 5, 3, 64], stddev=5e-2, w1=0.0)
kernel1 = tf.nn.conv2d(image_holder, weight1, [1, 1, 1, 1], padding='SAME')
bias1 = tf.Variable(tf.constant(0.0, shape=[64]))
conv1 = tf.nn.relu(tf.nn.bias_add(kernel1, bias1))
pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
# 创建第二个卷积层
weight2 = variable_with_weight_loss(shape=[5, 5, 64, 64], stddev=5e-2, w1=0.0)
kernel2 = tf.nn.conv2d(pool1, weight2, [1, 1, 1, 1], padding='SAME')
bias2 = tf.Variable(tf.constant(0.1, shape=[64]))
conv2 = tf.nn.relu(tf.nn.bias_add(kernel2, bias2))
pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
# 创建全连接层
reshape = tf.reshape(pool2, [batch_size, -1])
dim = reshape.get_shape()[1].value
weight3 = variable_with_weight_loss(shape=[dim, 384], stddev=0.04, w1=0.004)
bias3 = tf.Variable(tf.constant(0.1, shape=[384]))
local3 = tf.nn.relu(tf.matmul(reshape, weight3) + bias3)

weight4 = variable_with_weight_loss(shape=[384, 192], stddev=0.04, w1=0.004)
bias4 = tf.Variable(tf.constant(0.1, shape=[192]))
local4 = tf.nn.relu(tf.matmul(local3, weight4) + bias4)

weight5 = variable_with_weight_loss(shape=[192, 2], stddev=1/192.0, w1=0.0)
bias5 = tf.Variable(tf.constant(0.0, shape=[2]))
logits = tf.add(tf.matmul(local4, weight5), bias5)

# 解码过程

# 计算损失函数
def loss(logits, labels):
    labels = tf.cast(labels, tf.int32)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels, name='cross_entropy_per_example'
    )
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)
    return tf.add_n(tf.get_collection('losses'), name='total_loss'), cross_entropy


def get_accuracy(logits, labels):
    acc = tf.nn.in_top_k(logits, labels, 1)
    acc = tf.cast(acc, tf.float32)
    acc = tf.reduce_mean(acc)
    return acc

loss, cor = loss(logits, label_holder)

train_op = tf.train.AdamOptimizer(0.5).minimize(loss)
acc = get_accuracy(logits, label_holder)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
# 启动数据增强的队列线程
tf.train.start_queue_runners()

for step in range(max_step):
    start_time = time.time()
    image_batch, label_batch = sess.run([images_train, labels_train])
    _, loss_value, acc_value = sess.run([train_op, loss, acc],
                                        feed_dict={image_holder: image_batch, label_holder: label_batch})
    duration = time.time() - start_time
    if step % 10 == 0:
        examples_per_sec = batch_size / duration
        sec_per_batch = float(duration)

        format_str = ('step %d, loss=%.6f acc=%.6f (%.1f examples / sec; %.3f sec / batch)')
        print(format_str % (step, loss_value, acc_value, examples_per_sec, sec_per_batch))
        print(cor)
        print(label_batch)
num_examples = 12500

num_iter = int(math.ceil(num_examples / batch_size))
true_count = 0
total_sample_count = num_iter * batch_size
step = 0
acc_mean = 0
while step < num_iter:
    image_batch, label_batch = sess.run([images_test, labels_test])
    _, loss_value, acc_value = sess.run([train_op, loss, acc],
                                        feed_dict={image_holder: image_batch, label_holder: label_batch})
    step += 1
    if step % 10 == 0:
        print('step %d accuracy= %.6f' % (step, acc_value))


print("done")





