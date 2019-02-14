import random
import numpy as np
import os
import cv2
from os import scandir


def data_reader(input_dir, img_type='.jpeg'):
    file_paths = []
    file_labels = []
    label = -1
    for img_fold in scandir(input_dir):
        label = label + 1
        for img_file in scandir(img_fold):
            if img_file.name.endswith(img_type) and img_file.is_file():
                file_paths.append(img_file.path)
                file_labels.append(label)
    return file_paths, file_labels


def get_target_batch(batch_size, image_width, image_height, data_path):
    file_paths, file_labels = data_reader(data_path, '.png')
    maxSize = len(file_paths)
    idx_list = random.sample(range(0, maxSize), batch_size)
    files = []
    labels = []
    images = []
    if batch_size == 0:
        batch_size = maxSize
        for i in range(batch_size):
            files.append(file_paths[i])
            labels.append(file_labels[i])
            image = cv2.imread(os.path.join(file_paths[i]))
            image = cv2.resize(image, (image_width, image_height))
            images.append(image)
    else:
        for i in range(batch_size):
            for j in range(maxSize):
                if idx_list[i] == j:
                    files.append(file_paths[j])
                    labels.append(file_labels[j])
                    image = cv2.imread(os.path.join(file_paths[j]))
                    image = cv2.resize(image, (image_width, image_height))
                    images.append(image)
    return images, idx_list, len(file_paths), labels


if __name__ == '__main__':
    # file_paths, file_labels=data_reader('/home/hit/liaijia/NEW/datasets/toxo40x/train', img_type='.png')
    # print(file_paths)
    print(get_target_batch(0, 224, 224, "/home/hit/liaijia/NEW/datasets/toxo40x/train"))
    # print (len(file_paths))
    # print(file_labels)
