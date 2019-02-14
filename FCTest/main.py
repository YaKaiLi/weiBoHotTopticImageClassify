from network import vgg_network, siamese_loss, get_accuracy, get_nearst_accuracy
from resnet import res_network
import tensorflow as tf
import numpy as np
import os
from dataset.plas import get_siamese_batch, get_siamese_test_batch
from fuzzcyNet import fuzzyLoss, initialise_U, getSubU
from build_data import get_target_batch
from FCM import fuzzyCluster, fuzzy

os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'

batch_size = 50
test_batch = 100
cluster_num = 2
DIM = 100
M = 10000.0
mi = 2
UsavePath = 'numpy/U.txt'
image_width = 224
image_height = 224
channel = 3
rate = 0.5
learning_rate = 0.0001
train_step = 10000
save_path = "./siamese_malaria"    # 模型1识别率0.36
trainMethod = 'pre'  # pre-预训练, formal训练
#pre是使用孪生
#target是使用resNet+FCM

data_path = "/home/hit/liaijia/NEW/datasets/toxo40x/train"  # toxo_train_40x
# data_path = "/home/hit/liaijia/NEW/datasets/malaria/train"  # malaria


def train(data_path):
    image_1 = tf.placeholder(tf.float32, [None, image_width, image_height, channel], name="image_1")
    result_1 = res_network(image_1)

    image_2 = tf.placeholder(tf.float32, [None, image_width, image_height, channel], name="image_2")
    label = tf.placeholder(tf.float32, [None], name="label")
    result_2 = res_network(image_2)

    if trainMethod == 'pre':
        loss = siamese_loss(result_1, result_2, label)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
    else:
        tensor_u = tf.placeholder(tf.float32, [None, cluster_num])
        clusters = tf.placeholder(tf.float32, [cluster_num, DIM])
        loss = fuzzyLoss(data=result_1, data_n=batch_size, U=tensor_u, clusters=clusters)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        if tf.gfile.Exists(os.path.join(save_path, "checkpoint")):
            print("Loading model...")
            dtn = tf.train.latest_checkpoint(save_path)
            saver.restore(sess, dtn)
        else:
            if not os.path.exists(save_path):
                os.makedirs(save_path)

        sess.graph.finalize()

        if trainMethod == 'formal':
            xt, xt_idx, data_n, label = get_target_batch(0, image_width, image_height, data_path)
            features = []

            for img in xt:
                feature = sess.run(result_1, feed_dict={image_1: [img]})
                features.append(feature[0])

            if not tf.gfile.Exists(os.path.join(save_path, "checkpoint")):
                print("new model")
                U = initialise_U(len(features), 2, M)
                cluster = fuzzyCluster(features, U, 2, mi)
                np.savetxt(save_path + '/C.txt', cluster, fmt="%.20f", delimiter=",")
                np.savetxt(save_path + '/U.txt', U, fmt="%.20f", delimiter=",")
            else:
                print("old_model")
                U = np.loadtxt(save_path + '/U.txt', delimiter=",")
                cluster = np.loadtxt(save_path + '/C.txt', delimiter=",")

        print("Nreuse=self.reuse, name=ow begin to train!!!")
        for i in range(train_step):
            if trainMethod == 'pre':
                xs_1, xs_2, ys, _, _, _, _ = get_siamese_batch(batch_size, rate, image_width, image_height)
                _, get_loss = sess.run([optimizer, loss], feed_dict={image_1: xs_1, image_2: xs_2, label: ys})
            else:
                xt, xt_idx, data_n, _ = get_target_batch(batch_size, image_width, image_height, data_path)
                subU = getSubU(U, xt_idx)
                _, get_loss = sess.run([optimizer, loss], feed_dict={image_1: xt, clusters: cluster, tensor_u: subU})
                print("Step %d: loss: %.20f" % (i, get_loss))
                #初始化参数完成
                if (i+1) % 100 == 0:
                    xt, xt_idx, data_n, _ = get_target_batch(0, image_width, image_height, data_path)
                    features = []

                    for img in xt:
                        feature = sess.run(result_1, feed_dict={image_1: [img]})
                        features.append(feature[0])
                    U, cluster = fuzzy(data=features, U=U, cluster_number=cluster_num, m=mi)
                    np.savetxt(save_path + '/C.txt', cluster, fmt="%.20f", delimiter=",")
                    np.savetxt(save_path + '/U.txt', U, fmt="%.20f", delimiter=",")

            if (i+1) % 100 == 0:
                saver.save(sess, os.path.join(save_path, "ckpt"), global_step=i + 1)
                np.savetxt(save_path + '/C.txt', cluster, fmt="%.20f", delimiter=",")
                np.savetxt(save_path + '/U.txt', U, fmt="%.20f", delimiter=",")
                print("Saving model-%d step in" % (i+1), save_path)


def evaluate():
    with tf.Graph().as_default():
        image_1 = tf.placeholder(tf.float32, [None, image_width, image_height, channel], name="image_1")

        result_1 = vgg_network(image_1)

        saver = tf.train.Saver(tf.global_variables())
        with tf.Session() as sess:
            tf.global_variables_initializer().run()

            if tf.gfile.Exists(os.path.join(save_path, "checkpoint")):
                print("Loading model...")
                dtn = tf.train.latest_checkpoint(save_path)
                saver.restore(sess, dtn)

            xt, _, _, lt = get_target_batch(0, image_width, image_height, data_path)
            features = []
            for img in xt:
                feature = sess.run(result_1, feed_dict={image_1: [img]})
                features.append(feature[0])

            cluster = np.loadtxt(save_path + '/C.txt', delimiter=",")
            accuracy = get_nearst_accuracy(features, cluster, lt)
            print(accuracy)


def evaluate_siamese():
    with tf.Graph().as_default():
        image_1 = tf.placeholder(tf.float32, [None, image_width, image_height, channel], name="image_1")
        image_2 = tf.placeholder(tf.float32, [None, image_width, image_height, channel], name="image_2")

        result_1 = vgg_network(image_1)
        result_2 = vgg_network(image_2)

        saver = tf.train.Saver(tf.global_variables())
        with tf.Session() as sess:
            tf.global_variables_initializer().run()

            if tf.gfile.Exists(os.path.join(save_path, "checkpoint")):
                print("Loading model...")
                dtn = tf.train.latest_checkpoint(save_path)
                saver.restore(sess, dtn)

            xs_1, xs_2, l_1, l_2 = get_siamese_test_batch(image_width, image_height)
            feature_1 = []
            feature_2 = []
            for i in range(len(l_1) // test_batch):
                feature = sess.run(result_1, feed_dict={image_1: xs_1[i * test_batch: i * test_batch + test_batch]})
                feature_1[i * test_batch:] = feature
            for i in range(len(l_2) // test_batch):
                feature = sess.run(result_2, feed_dict={image_2: xs_2[i * test_batch: i * test_batch + test_batch]})
                feature_2[i * test_batch:] = feature
            accuracy = get_accuracy(feature_2, feature_1, l_2, l_1)
            print(accuracy)


if __name__ == '__main__':
    train(data_path)
    # evaluate()
    # evaluate_siamese()
