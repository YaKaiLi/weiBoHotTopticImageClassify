# coding=utf-8
import tensorflow as tf
import random
import numpy as np
import time
import sys
import matplotlib.pyplot as plt
import decimal


# 这里初始化的应该为所有样本的

def initialise_U(data_n, cluster_number, MAX):
    """
    这个函数是隶属度矩阵U的每行加起来都为1. 此处需要一个全局变量MAX.
    """
    U = []
    for i in range(0, data_n):
        current = []
        rand_sum = 0.0
        for j in range(0, cluster_number):
            dummy = random.randint(1, int(MAX))
            current.append(dummy)
            rand_sum += dummy
        for j in range(0, cluster_number):
            current[j] = current[j] / rand_sum
        U.append(current)
    return U


def getSubU(U, idx):
    target = []
    # print('U', type(U))
    # print('idx', type(idx))
    for i in idx:
        # print(i)
        # print('U[j]', U[i])
        target.append(U[i])

    # print('target', target)
    return target


def coverU(U, idx, subU):
    for i in range(len(subU)):
        for j in range(len(U)):
            if idx[i] == j:
                U[idx[i]] = subU[i]
    return U


def fuzzy_batch(data, U, idx, data_n, DIM, cluster_num, M):
    subU = getSubU(U, idx)
    data_feed = tf.placeholder(tf.float32, [data_n, DIM])
    tensor_m = tf.constant(np.ones([data_n, cluster_num], dtype=np.float32) * M)
    tensor_U = tf.constant(subU)
    # U = initialise_U(Batch_size, Cluster_num)
    # print('U', tensor_U)
    print('正在执行第---{}---迭代计算', iter)
    UM = tf.pow(tensor_U, tensor_m)
    dumpy_sum_num = tf.matmul(UM, data_feed, transpose_a=True)
    dum = tf.expand_dims(tf.reduce_sum(UM, 0), 1)
    g = []
    for i in range(DIM):
        g.append(dum)
    dumpy_sum_dum = tf.concat(g, axis=1)
    clusters = tf.divide(dumpy_sum_num, dumpy_sum_dum)
    c1 = []
    c2 = []
    for i in range(data_n):
        c1.append(tf.expand_dims(clusters[0], 0))
        c2.append(tf.expand_dims(clusters[1], 0))
    cluster_1 = tf.concat(c1, axis=0)
    cluster_2 = tf.concat(c2, axis=0)
    distance_1 = tf.reduce_mean(tf.sqrt(tf.pow(tf.subtract(data_feed, cluster_1), 2)), axis=1)
    distance_2 = tf.reduce_mean(tf.sqrt(tf.pow(tf.subtract(data_feed, cluster_2), 2)), axis=1)
    distance = tf.concat([tf.expand_dims(distance_1, 1), tf.expand_dims(distance_2, 1)], axis=1)
    distance_ = tf.concat([tf.expand_dims(distance_2, 1), tf.expand_dims(distance_1, 1)], axis=1)

    tensor_ones = tf.ones(shape=[data_n, cluster_num])
    tensor_twos = tf.multiply(tensor_ones, 2)
    tensor_mi = tf.divide(tensor_twos, tf.subtract(tensor_ones, tensor_m))  # 2/-(m-1)=2/1-m
    dist_dist_i = tf.pow(tf.divide(distance, distance), tensor_mi)
    dist_dist_j = tf.pow(tf.divide(distance, distance_), tensor_mi)

    dist_dist = dist_dist_i + dist_dist_j

    tf_U = tf.divide(tensor_ones, dist_dist)
    with tf.Session() as sess:
        [target, clusters_] = sess.run([tf_U, clusters], feed_dict={data_feed: data})
    targetU = coverU(U, idx, target)
    return targetU, clusters_


def fuzzy(data, U, data_n, DIM, cluster_num, M):
    data_feed = tf.placeholder(tf.float32, [data_n, DIM])
    tensor_m = tf.constant(np.ones([data_n, cluster_num], dtype=np.float32) * M)
    tensor_U = tf.constant(U)
    # U = initialise_U(Batch_size, Cluster_num)
    # print('U', tensor_U)
    print('正在执行第---{}---迭代计算', iter)
    UM = tf.pow(tensor_U, tensor_m)
    dumpy_sum_num = tf.matmul(UM, data_feed, transpose_a=True)
    dum = tf.expand_dims(tf.reduce_sum(UM, 0), 1)
    g = []
    for i in range(DIM):
        g.append(dum)
    dumpy_sum_dum = tf.concat(g, axis=1)
    clusters = tf.divide(dumpy_sum_num, dumpy_sum_dum)
    c1 = []
    c2 = []
    for i in range(data_n):
        c1.append(tf.expand_dims(clusters[0], 0))
        c2.append(tf.expand_dims(clusters[1], 0))
    cluster_1 = tf.concat(c1, axis=0)
    cluster_2 = tf.concat(c2, axis=0)
    print("clusters", clusters)
    print("c1", c1)
    print("cluster_1", cluster_1)
    distance_1 = tf.reduce_mean(tf.sqrt(tf.pow(tf.subtract(data_feed, cluster_1), 2)), axis=1)
    distance_2 = tf.reduce_mean(tf.sqrt(tf.pow(tf.subtract(data_feed, cluster_2), 2)), axis=1)
    distance = tf.concat([tf.expand_dims(distance_1, 1), tf.expand_dims(distance_2, 1)], axis=1)
    distance_ = tf.concat([tf.expand_dims(distance_2, 1), tf.expand_dims(distance_1, 1)], axis=1)

    tensor_ones = tf.ones(shape=[data_n, cluster_num])
    tensor_twos = tf.multiply(tensor_ones, 2)
    tensor_mi = tf.divide(tensor_twos, tf.subtract(tensor_ones, tensor_m))  # 2/-(m-1)=2/1-m
    dist_dist_i = tf.pow(tf.divide(distance, distance), tensor_mi)
    dist_dist_j = tf.pow(tf.divide(distance, distance_), tensor_mi)

    dist_dist = dist_dist_i + dist_dist_j

    tf_U = tf.divide(tensor_ones, dist_dist)
    with tf.Session() as sess:
        U_, clusters_ = sess.run([tf_U, clusters], feed_dict={data_feed: data})
    return U_, clusters_


def fuzzy(U, data_n, DIM, cluster_num, M):
    data_feed = tf.placeholder(tf.float32, [data_n, DIM])
    tensor_m = tf.constant(np.ones([data_n, cluster_num], dtype=np.float32) * M)
    tensor_U = tf.constant(U)
    # U = initialise_U(Batch_size, Cluster_num)
    # print('U', tensor_U)
    print('正在执行第---{}---迭代计算', iter)
    UM = tf.pow(tensor_U, tensor_m)
    dumpy_sum_num = tf.matmul(UM, data_feed, transpose_a=True)
    dum = tf.expand_dims(tf.reduce_sum(UM, 0), 1)
    g = []
    for i in range(DIM):
        g.append(dum)
    dumpy_sum_dum = tf.concat(g, axis=1)
    clusters = tf.divide(dumpy_sum_num, dumpy_sum_dum)
    return fuzzyLoss


def fuzzyLoss(data, data_n, U, clusters):
    c1 = []
    c2 = []
    # data_n=data.get_shape()
    # print('data_n shape len', data_n)
    for i in range(data_n):
        c1.append(tf.expand_dims(clusters[0], 0))
        c2.append(tf.expand_dims(clusters[1], 0))
    cluster_1 = tf.concat(c1, axis=0)
    cluster_2 = tf.concat(c2, axis=0)

    distance_1 = tf.reduce_mean(tf.sqrt(tf.pow(tf.subtract(data, cluster_1), 2)), axis=1)
    distance_2 = tf.reduce_mean(tf.sqrt(tf.pow(tf.subtract(data, cluster_2), 2)), axis=1)
    distance = tf.concat([tf.expand_dims(distance_1, 1), tf.expand_dims(distance_2, 1)], axis=1)
    fuzzyLoss = tf.reduce_mean(tf.multiply(distance, U))

    return fuzzyLoss


def normalise_U(U):
    """
    在聚类结束时使U模糊化。每个样本的隶属度最大的为1，其余为0
    """
    for i in range(0, len(U)):
        maximum = max(U[i])
        for j in range(0, len(U[0])):
            if U[i][j] != maximum:
                U[i][j] = 0
            else:
                U[i][j] = 1
    return U


def import_data_format_iris(file):
    """
    格式化数据，前四列为data，最后一列为cluster_location
    数据地址 http://archive.ics.uci.edu/ml/machine-learning-databases/iris/
    """
    data = []
    cluster_location = []
    with open(str(file), 'r') as f:
        for line in f:
            current = line.strip().split(",")
            current_dummy = []
            for j in range(0, len(current) - 1):
                current_dummy.append(float(current[j]))
            j += 1
            if current[j] == "Iris-setosa\n":
                cluster_location.append(0)
            elif current[j] == "Iris-versicolor\n":
                cluster_location.append(1)
            else:
                cluster_location.append(2)
            data.append(current_dummy)
    print("加载数据完毕")
    return data, cluster_location


def randomise_data(data):
    """
    该功能将数据随机化，并保持随机化顺序的记录
    """
    order = list(range(0, len(data)))
    random.shuffle(order)
    new_data = [[] for i in range(0, len(data))]
    for index in range(0, len(order)):
        new_data[index] = data[order[index]]
    return new_data, order


def de_randomise_data(data, order):
    """
    此函数将返回数据的原始顺序，将randomise_data()返回的order列表作为参数
    """
    new_data = [[] for i in range(0, len(data))]
    for index in range(len(order)):
        new_data[order[index]] = data[index]
    return new_data


def checker_iris(final_location):
    """
	和真实的聚类结果进行校验比对
	"""
    right = 0.0
    for k in range(0, 3):
        checker = [0, 0, 0]
        for i in range(0, 50):
            for j in range(0, len(final_location[0])):
                if final_location[i + (50 * k)][j] == 1:
                    checker[j] += 1
        right += max(checker)
        print(right)
    answer = right / 150 * 100
    return "准确度：" + str(answer) + "%"


if __name__ == '__main__':
    data_feed, cluster_location = import_data_format_iris("iris.txt")
    data_feed, order = randomise_data(data_feed)

    U = initialise_U(len(data_feed), 2, 10000.0)

    target, cluster = fuzzy(data_feed, U, len(data_feed), 4, 2, 2)
    target = normalise_U(target)
    data = tf.placeholder(tf.float32, [150, 4])

    print('target', target)
    print('cluster', cluster)
    final_location = de_randomise_data(target, order)
    print(checker_iris(final_location))
    loss = fuzzyLoss(data, target, cluster)
    with tf.Session() as sess:
        loss_ = sess.run([loss], feed_dict={data: data_feed})
    print('loss', loss_)
