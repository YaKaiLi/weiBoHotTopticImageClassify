from network import vgg_network
import tensorflow as tf
import numpy as np
import os
from fuzzcyNet import initialise_U
from build_data import get_target_batch
from FCM import fuzzyCluster
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

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
domain = 'target'  # source, target, 疟疾， 弓形虫

data_path = "/home/hit/liaijia/NEW/datasets/toxo40x/train"  # toxo_train_40x


def train(data_path):
    image_1 = tf.placeholder(tf.float32, [None, image_width, image_height, channel], name="image_1")
    result_1 = vgg_network(image_1)

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

        if domain == 'target':
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
            #     U = np.loadtxt(save_path + '/U.txt', delimiter=",")
            #     cluster = np.loadtxt(save_path + '/C.txt', delimiter=",")

            print("Nreuse=self.reuse, name=ow begin to train!!!")

            xt, xt_idx, data_n, lll = get_target_batch(0, image_width, image_height, data_path)
            print(xt_idx)
            print(xt)

            features = []
            for img in xt:
                feature = sess.run(result_1, feed_dict={image_1: [img]})
                print(feature[0])
                features.append(feature[0])
            # cluster = np.loadtxt(save_path + '/C.txt', delimiter=",")
            print(np.shape(features))

            tsne = TSNE(n_components=2, init="pca", random_state=0)
            X_tsne = tsne.fit_transform(features)
            X_tsne = list(X_tsne)
            print(X_tsne)

            x1 = []
            x2 = []
            y1 = []
            y2 = []
            for index in range(len(X_tsne)):
                if lll[index] == 1:
                    x1.append(X_tsne[index][0])
                    y1.append(X_tsne[index][1])
                else:
                    x2.append(X_tsne[index][0])
                    y2.append(X_tsne[index][1])

            plt.plot(x1, y1, "ro", x2, y2, "b^")
            plt.xticks(np.arange(-40, 60, 10))
            plt.yticks(np.arange(-25, 40, 5))
            plt.show()


if __name__ == '__main__':
    train(data_path)

