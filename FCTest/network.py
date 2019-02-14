import tensorflow as tf
import tensorflow.contrib as cb
import numpy as np

weight_decay = 0.0005
margin = 3.0
alpha = 0.1


def convolution(tensor_in, filters, name=None):
    with tf.variable_scope("convolution", reuse=tf.AUTO_REUSE):
        return tf.layers.conv2d(tensor_in, filters, [3, 3], activation=tf.nn.relu,
                                kernel_regularizer=cb.layers.l2_regularizer(weight_decay), name=name)


def pool(tensor_in, name):
    with tf.variable_scope("pool", reuse=tf.AUTO_REUSE):
        return tf.layers.max_pooling2d(tensor_in, [2, 2], [2, 2], name=name)


def vgg_network(tensor_in):
    with tf.variable_scope("vgg_network", reuse=tf.AUTO_REUSE):

        conv_1 = convolution(tensor_in, 64, "conv_1")
        conv_2 = convolution(conv_1, 64, "conv_2")
        pool_1 = pool(conv_2, "pool_1")
        print("pool_1", pool_1)

        conv_3 = convolution(pool_1, 128, "conv_3")
        conv_4 = convolution(conv_3, 128, "conv_4")
        pool_2 = pool(conv_4, "pool_2")
        print("pool_2", pool_2)

        conv_5 = convolution(pool_2, 256, "conv_5")
        conv_6 = convolution(conv_5, 256, "conv_6")
        conv_7 = convolution(conv_6, 256, "conv_7")
        pool_3 = pool(conv_7, "pool_3")
        print("pool_3", pool_3)

        conv_8 = convolution(pool_3, 512, "conv_8")
        conv_9 = convolution(conv_8, 512, "conv_9")
        conv_10 = convolution(conv_9, 512, "conv_10")
        pool_4 = pool(conv_10, "pool_4")
        print("pool_4", pool_4)

        conv_11 = convolution(pool_4, 512, "conv_11")
        conv_12 = convolution(conv_11, 512, "conv_12")
        conv_13 = convolution(conv_12, 512, "conv_13")
        pool_5 = pool(conv_13, "pool_5")
        print("pool_5", pool_5)

        pool_shape = pool_5.get_shape().as_list()
        nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
        reshape = tf.reshape(pool_5, [-1, nodes])
        print("reshape", reshape)

        dense_1 = tf.layers.dense(reshape, 4096, activation=tf.nn.relu,
                                  kernel_regularizer=cb.layers.l2_regularizer(weight_decay))
        dense_2 = tf.layers.dense(dense_1, 4096, activation=tf.nn.relu,
                                  kernel_regularizer=cb.layers.l2_regularizer(weight_decay))
        dense_3 = tf.layers.dense(dense_2, 100, activation=tf.nn.relu,
                                  kernel_regularizer=cb.layers.l2_regularizer(weight_decay))
        print("dense_3", dense_3)

        return dense_3


def siamese_loss(feature_1, feature_2, label):
    label_fake = tf.subtract(1.0, label, name="label_fake")

    euc_distance = tf.square(tf.subtract(feature_1, feature_2))
    euc_distance = tf.reduce_sum(euc_distance, 1)
    print(euc_distance)

    margin_s = tf.constant(margin, name="margin")
    euc_distance_s = tf.sqrt(euc_distance+1e-6, name="euc_distance_s")
    positive = tf.multiply(label, euc_distance, name="positive")
    negative = tf.multiply(label_fake, tf.square(tf.maximum(tf.subtract(margin_s, euc_distance_s), 0)), name="negative")
    loss = tf.reduce_mean(tf.add(positive, negative), name="loss")

    return loss


def get_accuracy(f_test, f_train, l_test, l_train):
    # f_test测试集特征 f_train训练集特征 l_test l_train测试和训练集label
    test_num = len(f_test)
    accuracy = 0    # 判断结果正确的测试样本数量
    print("test Number:", test_num, "samples In Computing Accuracy!")

    for test in range(test_num):
        a_test = f_test[test]   # 一个测试样本
        distance = []
        for train in range(len(f_train)):   # 和所有训练样本求距离
            a_train = f_train[train]
            distance.append(np.mean(np.square(a_test - a_train)))
        index = np.argmin(distance)  # 找出最小距离的train下标
        if l_test[test] == l_train[index]:
            accuracy += 1   # 如果test和index号train样本的label一致，则判断为正确

    return accuracy / test_num


def get_nearst_accuracy(datas, C, label):
    distances = []
    for data in datas:
        distance = []
        distance_1 = np.mean(np.sqrt(np.square(np.subtract(data, C[0]))))
        distance_2 = np.mean(np.sqrt(np.square(np.subtract(data, C[1]))))
        distance.append(distance_1)
        distance.append(distance_2)
        distances.append(distance)

    accuracy = 0
    for index in range(len(distances)):
        arg = np.argmin(distances[index])
        if arg == label[index]:
            accuracy += 1

    return accuracy / len(datas)