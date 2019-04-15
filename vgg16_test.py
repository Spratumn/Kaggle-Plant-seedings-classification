import cv2 as cv
import numpy as np
from model.VGG16 import VGG16
import tensorflow as tf
import time
import os


def get_file_name(file_dir, type_list):
    file_name_list = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] in type_list:
                file_name_list.append(root + '/' + file)
    return file_name_list


def load_image(image_path):
    img = cv.imread(image_path)
    img = cv.resize(img, (224, 224))
    return img


def get_prediction_label(prediction, file_path):
    label_list = np.array([l.strip() for l in open(file_path).readlines()])
    prediction_index = np.argsort(prediction)[::-1]
    label_top1 = label_list[prediction_index[0]]
    label_top5 = label_list[prediction_index[0:5]]
    return label_top1, label_top5


def test(images_name, batch):
    batch_size = len(batch_image)
    with tf.Session() as sess:
        images = tf.placeholder("float", [batch_size, 224, 224, 3])
        feed_dict = {images: batch}
        vgg = VGG16(training=False)
        vgg.forward_propagation(images)
        print('predict start...')
        start = time.time()
        prediction = tf.nn.softmax(vgg.output)
        batch_prediction = sess.run(prediction, feed_dict=feed_dict)
        print('compute finished')
        end = time.time()
        cost = end - start
        print('compute cost {} s'.format(cost))
        for i in range(batch_size):
            top1, top5 = get_prediction_label(batch_prediction[i], 'model/synset.txt')
            image_name = images_name[i].split('/')[-1]
            print('image name:', image_name, 'prediction: ', top1)


if __name__ == '__main__':
    file_name = get_file_name('./test_data', ['.png', '.jpeg', '.jpg'])

    image_list = [load_image(file_path) for file_path in file_name]

    batch_image = np.array(image_list)

    test(file_name, batch_image)




