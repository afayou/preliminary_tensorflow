import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

slim = tf.contrib.slim


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def identity_block(X_input, kernel_size, in_filter, out_filters, stage, block, training):
    # defining name basis
    block_name = 'res' + str(stage) + block
    f1, f2, f3 = out_filters
    with tf.variable_scope(block_name):
        X_shortcut = X_input

        # first
        W_conv1 = weight_variable([1, 1, in_filter, f1])
        X = tf.nn.conv2d(X_input, W_conv1, strides=[1, 1, 1, 1], padding='SAME')
        X = tf.layers.batch_normalization(X, axis=3, training=training)
        X = tf.nn.relu(X)

        # second
        W_conv2 = weight_variable([kernel_size, kernel_size, f1, f2])
        X = tf.nn.conv2d(X, W_conv2, strides=[1, 1, 1, 1], padding='SAME')
        X = tf.layers.batch_normalization(X, axis=3, training=training)
        X = tf.nn.relu(X)

        # third

        W_conv3 = weight_variable([1, 1, f2, f3])
        X = tf.nn.conv2d(X, W_conv3, strides=[1, 1, 1, 1], padding='VALID')
        X = tf.layers.batch_normalization(X, axis=3, training=training)

        # final step
        add = tf.add(X, X_shortcut)
        add_result = tf.nn.relu(add)

    return add_result

def convolutional_block(X_input, kernel_size, in_filter,
                            out_filters, stage, block, training, stride=2):
    # defining name basis
    block_name = 'res' + str(stage) + block
    with tf.variable_scope(block_name):
        f1, f2, f3 = out_filters

        x_shortcut = X_input
        #first
        W_conv1 = weight_variable([1, 1, in_filter, f1])
        X = tf.nn.conv2d(X_input, W_conv1,strides=[1, stride, stride, 1],padding='VALID')
        X = tf.layers.batch_normalization(X, axis=3, training=training)
        X = tf.nn.relu(X)

        #second
        W_conv2 = weight_variable([kernel_size, kernel_size, f1, f2])
        X = tf.nn.conv2d(X, W_conv2, strides=[1,1,1,1], padding='SAME')
        X = tf.layers.batch_normalization(X, axis=3, training=training)
        X = tf.nn.relu(X)

        #third
        W_conv3 = weight_variable([1,1, f2,f3])
        X = tf.nn.conv2d(X, W_conv3, strides=[1, 1, 1,1], padding='VALID')
        X = tf.layers.batch_normalization(X, axis=3, training=training)

        #shortcut path
        W_shortcut = weight_variable([1, 1, in_filter, f3])
        x_shortcut = tf.nn.conv2d(x_shortcut, W_shortcut, strides=[1, stride, stride, 1], padding='VALID')

        #final
        add = tf.add(x_shortcut, X)
        add_result = tf.nn.relu(add)

    return add_result

def deepnn(x_input):

    x = tf.pad(x_input, tf.constant([[0, 0], [3, 3], [3, 3], [0, 0]]), "CONSTANT")
    with tf.variable_scope('reference'):
        training = tf.placeholder(tf.bool, name='training')

        #stage 1
        w_conv1 = weight_variable([7, 7, 3, 64])
        x = tf.nn.conv2d(x, w_conv1, strides=[1, 2, 2, 1], padding='VALID')
        x = tf.layers.batch_normalization(x, axis=3, training=training)
        x = tf.nn.relu(x)
        x = tf.nn.max_pool(x, ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1], padding='VALID')
        # assert (x.get_shape() == (x.get_shape()[0], 15, 15, 64))

        #stage 2
        x = convolutional_block(x, 3, 64, [64, 64, 256], 2, 'a', training, stride=1)
        x = identity_block(x, 3, 256, [64, 64, 256], stage=2, block='b', training=training)
        x = identity_block(x, 3, 256, [64, 64, 256], stage=2, block='c', training=training)

        #stage 3
        x = convolutional_block(x, 3, 256, [128, 128,512], 3, 'a', training)
        x = identity_block(x, 3, 512, [128, 128, 512], 3, 'b', training=training)
        x = identity_block(x, 3, 512, [128, 128, 512], 3, 'c', training=training)
        x = identity_block(x, 3, 512, [128, 128, 512], 3, 'd', training=training)

        #stage 4
        x = convolutional_block(x, 3, 512, [256, 256, 1024], 4, 'a', training)
        x = identity_block(x, 3, 1024, [256, 256, 1024], 4, 'b', training=training)
        x = identity_block(x, 3, 1024, [256, 256, 1024], 4, 'c', training=training)
        x = identity_block(x, 3, 1024, [256, 256, 1024], 4, 'd', training=training)
        x = identity_block(x, 3, 1024, [256, 256, 1024], 4, 'e', training=training)
        x = identity_block(x, 3, 1024, [256, 256, 1024], 4, 'f', training=training)

        #stage 5
        x = convolutional_block(x, 3, 1024, [512, 512, 2048], 5, 'a', training)
        x = identity_block(x, 3, 2048, [512, 512, 2048], 5, 'b', training=training)
        x = identity_block(x, 3, 2048, [512, 512, 2048], 5, 'c', training=training)

        x = tf.nn.avg_pool(x, [1, 2, 2, 1], strides=[1, 1, 1, 1], padding='VALID')

        flatten = tf.layers.flatten(x)
        x = tf.layers.dense(flatten, units=50, activation=tf.nn.relu)
        # Dropout - controls the complexity of the model, prevents co-adaptation of
        # features.
        with tf.name_scope('dropout'):
            keep_prob = tf.placeholder(tf.float32)
            x = tf.nn.dropout(x, keep_prob)

        logits = tf.layers.dense(x, units=6, activation=tf.nn.softmax)

    return logits, keep_prob, training
def cost(logits, labels):
    with tf.name_scope('loss'):
        # cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=y_conv)
        cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)
    cross_entropy_cost = tf.reduce_mean(cross_entropy)
    return cross_entropy_cost

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

batch_size = 32
learning_rate = 0.003
learning_rate_decay = 0.97
# regularization_rate = 0.0001
model_save_path = './model/'


def train():
    images = tf.placeholder(tf.float32, [None, 28, 28, 3])
    labels = tf.placeholder(tf.float32, [None, 10])
    global_step = tf.Variable(0, trainable=False)

    logits, keep_prob, train_mode = deepnn(images)

    cross_entropy = cost(logits, labels)

    with tf.name_scope('adam_optimizer'):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy, global_step=global_step)


    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(500):
            x_b, y_b = mnist.train.next_batch(batch_size)
            train_op_, loss_, step = sess.run([train_op, cross_entropy, global_step],
                                              feed_dict={images: x_b, labels: y_b})
            if i % 100 == 0:
                print("training step {0}, loss {1}".format(step, loss_))
            saver.save(sess, model_save_path + 'my_model', global_step=global_step)

def main(_):
    train()


if __name__ == '__main__':
    tf.app.run()


