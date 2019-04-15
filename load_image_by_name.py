import os
import cv2 as cv
import numpy as np
from random import randint
from sklearn.preprocessing import OneHotEncoder
"""
使用时
from load_image_by_name import get_name_label_set, load_batch_images

get_name_label_set(father_path, class_name_list, split_rate=0.9, hot=True)
参数：father_path，分类存放的图片上一级路径
      class_name_list， 分类存放的各类文件夹名称
      split_rate， 拆分训练集与测试集的比率
      hot, 是否进行独热编码
返回值：train_names, train_labels, test_names, test_labels
      train_names， 训练集的图片路径列表
      train_labels，与训练集图片路径对应的图片真实独热标签
      test_names， 测试集的图片路径列表
      test_labels，与测试集图片路径对应的图片真实独热标签
      
load_batch_images(names, labels, image_width, batch_size)
参数：names，上面函数得到的训练集或测试集图片路径列表
      labels， 与图片路径列表对应的图片真实独热标签
      image_width， 加载图片时控制图片尺寸
      batch_size，加载图片数量
返回值：image_batch, y_batch
      image_batch, 一个训练或测试的图片batch
      y_batch， 与图片batch对应的label
"""


def file_name(root_path, type_list):
    """
    可以遍历所有子目录
    :param root_path:
    :param type_list:
    :return:
    """
    file_name_list = []
    for root, _, files in os.walk(root_path):
        if files:
            for image_name in files:
                if image_name.split('.')[-1] in type_list:
                    path_split_list = root.split("\\")
                    image_path = ''
                    for name in path_split_list:
                        image_path += name
                        image_path += '/'
                    image_path += image_name
                    file_name_list.append(image_path)
                    print(image_path)
    return file_name_list


def get_file_name(father_path, type_list):
    """
    不能遍历子文件夹
    :param father_path:
    :param type_list:
    :return:
    """
    file_name_list = []
    for root, dirs, files in os.walk(father_path):
        for file in files:
            if os.path.splitext(file)[1] in type_list:
                file_name_list.append(root + '/' + file)
    return file_name_list


def load_image(image_path, image_width, rgb=False):
    image = cv.imread(image_path)
    image = cv.resize(image, (image_width, image_width))
    image_choice = randint(1, 4)
    if image_choice == 1:
        image = rotate_image(image)
    elif image_choice == 2:
        image = rotate_image(rotate_image(image))
    elif image_choice == 3:
        image = rotate_image(rotate_image(rotate_image(image)))
    elif image_choice == 4:
        image = flip_image(image)
    if rgb:
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    return image


def rotate_image(image):
    image_out = cv.transpose(image)
    image_out = cv.flip(image_out, 1)
    return image_out


def flip_image(image):
    image_out = cv.flip(image, 1)
    return image_out


def image_augment(root_path):
    type_list = ['png', 'jpeg', 'jpg']
    images_name_list = file_name(root_path, type_list)
    for image_name in images_name_list:
        image = cv.imread(image_name)
        _, image_path_name, image_type = image_name.split('.')

        image_v1 = rotate_image(image)
        image_v1_name = '.' + image_path_name + '_v1.' + image_type
        cv.imwrite(image_v1_name, image_v1)

        image_v2 = rotate_image(image_v1)
        image_v2_name = '.' + image_path_name + '_v2.' + image_type
        cv.imwrite(image_v2_name, image_v2)

        image_v3 = rotate_image(image_v2)
        image_v3_name = '.' + image_path_name + '_v3.' + image_type
        cv.imwrite(image_v3_name, image_v3)

        image_v4 = flip_image(image)
        image_v4_name = '.' + image_path_name + '_v4.' + image_type
        cv.imwrite(image_v4_name, image_v4)
    print('image augment finished.')


def get_name_label_set(father_path, class_name_list, split_rate=0.9, hot=True):
    file_names = []
    file_labels = []
    for i in range(len(class_name_list)):
        file_name_class1 = get_file_name(father_path + '/' + class_name_list[i], ['.png', '.jpeg', '.jpg'])
        file_names += file_name_class1
        file_labels += [i] * len(file_name_class1)
    randomly_index = np.random.permutation(len(file_labels))

    # 打乱顺序
    names_list = np.array(file_names)
    labels_list = np.array(file_labels)
    names_list = names_list[randomly_index]
    labels_list = labels_list[randomly_index]

    # 独热编码
    if hot:
        labels_list = OneHotEncoder(sparse=False
                                    , categories='auto').fit(labels_list.reshape(-1, 1)).transform(
            labels_list.reshape(-1, 1))

    # 拆分训练集
    cut_count = int(len(labels_list) * split_rate)
    train_names = names_list[0:cut_count]
    train_labels = labels_list[0:cut_count]
    test_names = names_list[cut_count:]
    test_labels = labels_list[cut_count:]
    return train_names, train_labels, test_names, test_labels


def load_batch_images(names, labels, image_width, batch_size):
    sample_num = len(labels)
    rand_index = np.random.choice(sample_num, size=batch_size)
    # print(rand_index)
    x_batch = names[rand_index]
    y_batch = labels[rand_index]
    image_list = [load_image(file_path, image_width) for file_path in x_batch]
    image_batch = np.array(image_list)
    return image_batch, y_batch

