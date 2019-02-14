import cv2
import os
import random

train_dataset = "/home/hit/liaijia/NEW/datasets/malaria/train"
test_dataset = "/home/hit/liaijia/NEW/datasets/malaria/test"


def get_all_images_path(data_type):
    if data_type == "train":
        path_dataset = train_dataset
    else:
        path_dataset = test_dataset
    datas = os.listdir(path_dataset)

    path_images = []
    cell = os.listdir(os.path.join(path_dataset, datas[0]))
    for image in cell:
        image = "0" + datas[0] + "/" + image
        path_images.append(image)
    plas = os.listdir(os.path.join(path_dataset, datas[1]))
    for image in plas:
        image = "1" + datas[1] + "/" + image
        path_images.append(image)
    return path_images


def read_image(path_image, image_width, image_height, data_type):
    if data_type == "train":
        path_dataset = train_dataset
    else:
        path_dataset = test_dataset

    label = int(path_image[0]) - 1
    image = cv2.imread(os.path.join(path_dataset, path_image[1:]))
    image = cv2.resize(image, (image_width, image_height))
    return image, label


def get_siamese_batch(batch_size, rate, image_width, image_height):
    images_1 = []
    images_2 = []
    path_images_1 = []
    path_images_2 = []
    labels = []
    labels_1 = []
    labels_2 = []

    all_images = get_all_images_path("train")
    positive_num = int(batch_size * rate)
    for batch in range(batch_size):
        path_random_image_1 = random.choice(all_images)
        image_1, label_1 = read_image(path_random_image_1, image_width, image_height, "train")
        if batch < positive_num:
            while True:
                path_random_image_2 = random.choice(all_images)
                image_2, label_2 = read_image(path_random_image_2, image_width, image_height, "train")
                if label_2 == label_1:
                    label = 1
                    break
        else:
            while True:
                path_random_image_2 = random.choice(all_images)
                image_2, label_2 = read_image(path_random_image_2, image_width, image_height, "train")
                if label_2 != label_1:
                    label = 0
                    break

        images_1.append(image_1)
        images_2.append(image_2)
        path_images_1.append(path_random_image_1)
        path_images_2.append(path_random_image_2)

        labels.append(label)
        label_i = [0] * 2
        label_i[label_1] = 1
        labels_1.append(label_i)
        label_i = [0] * 2
        label_i[label_2] = 1
        labels_2.append(label_i)

    return images_1, images_2, labels, labels_1, labels_2, path_images_1, path_images_2


def get_siamese_test_batch(image_width, image_height):
    train_images = []
    train_label = []
    test_images = []
    test_label = []

    all_images_train = get_all_images_path("train")
    for path_image in all_images_train:
        image, label = read_image(path_image, image_width, image_height, "train")
        train_images.append(image)
        train_label.append(label)

    all_images_test = get_all_images_path("test")
    for path_image in all_images_test:
        image, label = read_image(path_image, image_width, image_height, "test")
        test_images.append(image)
        test_label.append(label)

    return train_images, test_images, train_label, test_label


if __name__ == '__main__':
    print(get_siamese_batch(5, 0.5, 224, 224))

