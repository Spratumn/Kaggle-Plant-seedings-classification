import tensorflow as tf
import cv2 as cv
import os
import pickle
import time
from matplotlib import pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from random import randint
import scipy.io as sio


# *****************************************************************************************************
# define net class
class VGG16:
    def __init__(self, num_class=1000, istrain=False, model_path='model/imagenet-vgg-verydeep-16.mat'):
        if istrain:
            self.num_class = num_class
            with tf.variable_scope("conv1", reuse=tf.AUTO_REUSE):
                self.weight1_1 = self.weight(name='weight1_1', shape=[3, 3, 3, 64])
                self.biases1_1 = self.bias(shape=[64], name='biases1_1')
                self.weight1_2 = self.weight(name='weight1_2', shape=[3, 3, 64, 64])
                self.biases1_2 = self.bias(shape=[64], name='biases1_2')
            with tf.variable_scope("conv2", reuse=tf.AUTO_REUSE):
                self.weight2_1 = self.weight(name='weight2_1', shape=[3, 3, 64, 128])
                self.biases2_1 = self.bias(shape=[128], name='biases2_1')
                self.weight2_2 = self.weight(name='weight2_2', shape=[3, 3, 128, 128])
                self.biases2_2 = self.bias(shape=[128], name='biases2_2')
            with tf.variable_scope("conv3", reuse=tf.AUTO_REUSE):
                self.weight3_1 = self.weight(name='weight3_1', shape=[3, 3, 128, 256])
                self.biases3_1 = self.bias(shape=[256], name='biases3_1')
                self.weight3_2 = self.weight(name='weight3_2', shape=[3, 3, 256, 256])
                self.biases3_2 = self.bias(shape=[256], name='biases3_2')
                self.weight3_3 = self.weight(name='weight3_3', shape=[3, 3, 256, 256])
                self.biases3_3 = self.bias(shape=[256], name='biases3_3')
            with tf.variable_scope("conv4", reuse=tf.AUTO_REUSE):
                self.weight4_1 = self.weight(name='weight4_1', shape=[3, 3, 256, 512])
                self.biases4_1 = self.bias(shape=[512], name='biases4_1')
                self.weight4_2 = self.weight(name='weight4_2', shape=[3, 3, 512, 512])
                self.biases4_2 = self.bias(shape=[512], name='biases4_2')
                self.weight4_3 = self.weight(name='weight4_3', shape=[3, 3, 512, 512])
                self.biases4_3 = self.bias(shape=[512], name='biases4_3')
            with tf.variable_scope("conv5", reuse=tf.AUTO_REUSE):
                self.weight5_1 = self.weight(name='weight5_1', shape=[3, 3, 512, 512])
                self.biases5_1 = self.bias(shape=[512], name='biases5_1')
                self.weight5_2 = self.weight(name='weight5_2', shape=[3, 3, 512, 512])
                self.biases5_2 = self.bias(shape=[512], name='biases5_2')
                self.weight5_3 = self.weight(name='weight5_3', shape=[3, 3, 512, 512])
                self.biases5_3 = self.bias(shape=[512], name='biases5_3')
            with tf.variable_scope("full6", reuse=tf.AUTO_REUSE):
                self.weight6 = self.weight(name='weight6', shape=[7 * 7 * 512, 4096])
                self.biases6 = self.bias(shape=[4096], name='biases6')
            with tf.variable_scope("full7", reuse=tf.AUTO_REUSE):
                self.weight7 = self.weight(name='weight7', shape=[4096, 4096])
                self.biases7 = self.bias(shape=[4096], name='biases7')
            with tf.variable_scope("full8", reuse=tf.AUTO_REUSE):
                self.weight8 = self.weight(name='weight8', shape=[4096, self.num_class])
                self.biases8 = self.bias(shape=[self.num_class], name='biases8')
        else:
            self.load_data = sio.loadmat(model_path)
            self.weight1_1 = tf.constant(self.load_data['layers'][0][0][0][0][2][0][0], name="conv1_1_weights")
            self.biases1_1 = tf.constant(self.load_data['layers'][0][0][0][0][2][0][1].reshape([64])
                                         , name="conv1_1_biases")
            self.weight1_2 = tf.constant(self.load_data['layers'][0][2][0][0][2][0][0], name="conv1_2_weights")
            self.biases1_2 = tf.constant(self.load_data['layers'][0][2][0][0][2][0][1].reshape([64])
                                         , name="conv1_2_biases")

            self.weight2_1 = tf.constant(self.load_data['layers'][0][5][0][0][2][0][0], name="conv2_1_weights")
            self.biases2_1 = tf.constant(self.load_data['layers'][0][5][0][0][2][0][1].reshape([128])
                                         , name="conv2_1_biases")
            self.weight2_2 = tf.constant(self.load_data['layers'][0][7][0][0][2][0][0], name="conv2_2_weights")
            self.biases2_2 = tf.constant(self.load_data['layers'][0][7][0][0][2][0][1].reshape([128])
                                         , name="conv2_2_biases")

            self.weight3_1 = tf.constant(self.load_data['layers'][0][10][0][0][2][0][0], name="conv3_1_weights")
            self.biases3_1 = tf.constant(self.load_data['layers'][0][10][0][0][2][0][1].reshape([256])
                                         , name="conv3_1_biases")
            self.weight3_2 = tf.constant(self.load_data['layers'][0][12][0][0][2][0][0], name="conv3_2_weights")
            self.biases3_2 = tf.constant(self.load_data['layers'][0][12][0][0][2][0][1].reshape([256])
                                         , name="conv3_2_biases")
            self.weight3_3 = tf.constant(self.load_data['layers'][0][14][0][0][2][0][0], name="conv3_3_weights")
            self.biases3_3 = tf.constant(self.load_data['layers'][0][14][0][0][2][0][1].reshape([256])
                                         , name="conv3_3_biases")

            self.weight4_1 = tf.constant(self.load_data['layers'][0][17][0][0][2][0][0], name="conv4_1_weights")
            self.biases4_1 = tf.constant(self.load_data['layers'][0][17][0][0][2][0][1].reshape([512])
                                         , name="conv4_1_biases")
            self.weight4_2 = tf.constant(self.load_data['layers'][0][19][0][0][2][0][0], name="conv4_2_weights")
            self.biases4_2 = tf.constant(self.load_data['layers'][0][19][0][0][2][0][1].reshape([512])
                                         , name="conv4_2_biases")
            self.weight4_3 = tf.constant(self.load_data['layers'][0][21][0][0][2][0][0], name="conv4_3_weights")
            self.biases4_3 = tf.constant(self.load_data['layers'][0][21][0][0][2][0][1].reshape([512])
                                         , name="conv4_3_biases")

            self.weight5_1 = tf.constant(self.load_data['layers'][0][24][0][0][2][0][0], name="conv5_1_weights")
            self.biases5_1 = tf.constant(self.load_data['layers'][0][24][0][0][2][0][1].reshape([512])
                                         , name="conv5_1_biases")
            self.weight5_2 = tf.constant(self.load_data['layers'][0][26][0][0][2][0][0], name="conv5_2_weights")
            self.biases5_2 = tf.constant(self.load_data['layers'][0][26][0][0][2][0][1].reshape([512])
                                         , name="conv5_2_biases")
            self.weight5_3 = tf.constant(self.load_data['layers'][0][28][0][0][2][0][0], name="conv5_3_weights")
            self.biases5_3 = tf.constant(self.load_data['layers'][0][28][0][0][2][0][1].reshape([512])
                                         , name="conv5_3_biases")

            self.weight6 = tf.constant(self.load_data['layers'][0][31][0][0][2][0][0].reshape([7*7*512, 4096])
                                       , name="fc6_weights")
            self.biases6 = tf.constant(self.load_data['layers'][0][31][0][0][2][0][1].reshape([4096])
                                       , name="fc6_biases")

            self.weight7 = tf.constant(self.load_data['layers'][0][33][0][0][2][0][0].reshape([4096, 4096])
                                       , name="fc7_weights")
            self.biases7 = tf.constant(self.load_data['layers'][0][33][0][0][2][0][1].reshape([4096])
                                       , name="fc7_biases")

            self.weight8 = tf.constant(self.load_data['layers'][0][35][0][0][2][0][0].reshape([4096, 1000])
                                       , name="fc8_weights")
            self.biases8 = tf.constant(self.load_data['layers'][0][35][0][0][2][0][1].reshape([1000])
                                       , name="fc8_biases")

    def forward_propagation(self, images):
        relu1_1 = self.conv2d(images, self.weight1_1, self.biases1_1)
        relu1_2 = self.conv2d(relu1_1, self.weight1_2, self.biases1_2)
        pooling1 = self.maxpooling(relu1_2)  # shape(112,112,128)

        relu2_1 = self.conv2d(pooling1, self.weight2_1, self.biases2_1)  # shape(112,112,256)
        relu2_2 = self.conv2d(relu2_1, self.weight2_2, self.biases2_2)
        pooling2 = self.maxpooling(relu2_2)  # shape(56,56,128)

        relu3_1 = self.conv2d(pooling2, self.weight3_1, self.biases3_1)  # shape(56,56,256)
        relu3_2 = self.conv2d(relu3_1, self.weight3_2, self.biases3_2)
        relu3_3 = self.conv2d(relu3_2, self.weight3_3, self.biases3_3)
        pooling3 = self.maxpooling(relu3_3)  # shape(28,28,256)

        relu4_1 = self.conv2d(pooling3, self.weight4_1, self.biases4_1)  # shape(28,28,512)
        relu4_2 = self.conv2d(relu4_1, self.weight4_2, self.biases4_2)
        relu4_3 = self.conv2d(relu4_2, self.weight4_3, self.biases4_3)
        pooling4 = self.maxpooling(relu4_3)  # shape(14,14,512)

        relu5_1 = self.conv2d(pooling4, self.weight5_1, self.biases5_1)  # shape(14,14,512)
        relu5_2 = self.conv2d(relu5_1, self.weight5_2, self.biases5_2)
        relu5_3 = self.conv2d(relu5_2, self.weight5_3, self.biases5_3)
        self.pooling5 = self.maxpooling(relu5_3)  # shape(7,7,512)

        self.full_input6 = tf.reshape(self.pooling5, (-1, 7 * 7 * 512))  # shape(7 * 7 * 512)

        self.full_output6 = tf.nn.relu(tf.nn.bias_add(tf.matmul(self.full_input6, self.weight6), self.biases6))
        # shape(4096)
        self.full_output7 = tf.nn.relu(tf.nn.bias_add(tf.matmul(self.full_output6, self.weight7), self.biases7))
        # shape(4096)
        self.output = tf.nn.relu(tf.nn.bias_add(tf.matmul(self.full_output7, self.weight8), self.biases8))
        # shape(num_class)

    def weight(self, name, shape, dtype=tf.float32):
        initial = tf.truncated_normal_initializer(stddev=0.01)
        return tf.get_variable(initializer=initial, name=name, shape=shape, dtype=dtype
                               , collections=[tf.GraphKeys.GLOBAL_VARIABLES])

    def bias(self, name, shape, dtype=tf.float32):
        initial = tf.constant_initializer(0.0)
        return tf.get_variable(initializer=initial, name=name, shape=shape, dtype=dtype
                               , collections=[tf.GraphKeys.GLOBAL_VARIABLES])

    def conv2d(self, x, weight, biases, stride=1):
        # stride[1, x_movement, y_movement, 1]
        # Must have strides[0] = strides[3] =1
        # padding="SAME"用零填充边界
        conv = tf.nn.conv2d(x, weight, strides=[1, stride, stride, 1], padding="SAME")
        bias = tf.nn.bias_add(conv, biases)
        relu = tf.nn.relu(bias)
        return relu

    def maxpooling(self, x, ksize=2, stride=2):
        # stride[1, x_movement, y_movement, 1]
        # Must have strides[0] = strides[3] =1
        # padding="SAME"用零填充边界
        return tf.nn.max_pool(x, ksize=[1, ksize, ksize, 1], strides=[1, stride, stride, 1], padding="SAME")


# *****************************************************************************************************
# load data functions
def get_file_name(father_path, type_list):
    """
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


def rotate_image(image):
    image_out = cv.transpose(image)
    image_out = cv.flip(image_out, 1)
    return image_out


def flip_image(image):
    image_out = cv.flip(image, 1)
    return image_out


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


def get_name_label_set(father_path, class_name_list, split_rate=0.9, hot=True):
    file_names = []
    file_labels = []
    for i in range(len(class_name_list)):
        file_name_class1 = get_file_name(father_path + '/' + class_name_list[i], ['.png', '.jpeg', '.jpg'])
        file_names += file_name_class1
        file_labels += [i] * len(file_name_class1)
    randomly_index = np.random.permutation(len(file_labels))

    #
    names_list = np.array(file_names)
    labels_list = np.array(file_labels)
    names_list = names_list[randomly_index]
    labels_list = labels_list[randomly_index]

    # OneHotEncoder
    if hot:
        labels_list = OneHotEncoder(sparse=False
                                    , categories='auto').fit(labels_list.reshape(-1, 1)).transform(
            labels_list.reshape(-1, 1))

    # split train test
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


# *****************************************************************************************************
# model functions
def get_accuracy(predictions, labels):
    p = np.argmax(predictions, axis=1)
    o = np.argmax(labels, axis=1)
    crrect = np.equal(p, o)
    accuracy = 100. * ((crrect > 0).sum()) / len(crrect)
    return accuracy


def get_prediction_label(prediction, file_path, hastop5=False):
    label_list = np.array([l.strip() for l in open(file_path).readlines()])
    prediction_index = np.argsort(prediction)[::-1]
    label_top1 = label_list[prediction_index[0]]
    label_top5 = label_top1
    if hastop5:
        label_top5 = label_list[prediction_index[0:5]]
    return label_top1, label_top5


def plot_result(train_los, train_ac, test_ac):
    plt.figure('figure', figsize=(12, 5))
    ax1 = plt.subplot(1, 3, 1)
    ax2 = plt.subplot(1, 3, 2)
    ax3 = plt.subplot(1, 3, 3)
    plt.sca(ax1)
    plt.plot(range(len(train_los)), train_los, 'r--', label='Loss')
    plt.legend(loc='upper right', prop={'size': 11})
    plt.sca(ax2)
    plt.plot(range(len(train_ac)), train_ac, 'b--', label='train acc')
    plt.legend(loc='upper right', prop={'size': 11})
    plt.sca(ax3)
    plt.plot(range(len(train_ac)), test_ac, 'g--', label='test acc')
    plt.legend(loc='upper right', prop={'size': 11})
    plt.show()


# save model
def save_model(session, path):
    saver = tf.train.Saver(max_to_keep=4)
    saver.save(session, path)
    print('Trained Model Saved.')


def save_weights(weights_list):
    pickle.dump(weights_list, open('model/model_weights', 'wb'))
    print('weights Saved.')


# *****************************************************************************************************
# build net
class_list = ['Black-grass', 'Charlock', 'Cleavers', 'Common Chickweed'
              , 'Common wheat', 'Fat Hen', 'Loose Silky-bent', 'Maize'
              , 'Scentless Mayweed', 'Shepherds Purse', 'Small-flowered Cranesbill', 'Sugar beet']
train_names, train_labels, test_names, test_labels = get_name_label_set('./dataset/train', class_list)
print(len(train_names))


image_width = 224
batch_size = 32
epochs = 10
class_num = 12
new_train = False

# input batch images
X = tf.placeholder(tf.float32, shape=[batch_size, image_width, image_width, 3], name='X-input')
# real labels
labels = tf.placeholder(tf.float32, shape=[batch_size, class_num], name='labels-input')
learn_rate = tf.placeholder(tf.float16, name='learn_rate')

vgg = VGG16()
vgg.forward_propagation(X)

if new_train:
    with tf.variable_scope("full_layer", reuse=tf.AUTO_REUSE):
        weight6 = vgg.weight(name='weight6', shape=[7*7*512, 4096])
        biases6 = vgg.bias(shape=[4096], name='biases6')

        weight7 = vgg.weight(name='weight7', shape=[4096, 4096])
        biases7 = vgg.bias(shape=[4096], name='biases7')

        weight8 = vgg.weight(name='weight8', shape=[4096, class_num])
        biases8 = vgg.bias(shape=[class_num], name='biases8')
else:
    model_weights = pickle.load(open('model_weights', 'rb'))
    print(model_weights[0])

    # with tf.Session() as load_sess:
    #     saver = tf.train.import_meta_graph('./model/vgg.meta')
    #     saver.restore(load_sess, tf.train.latest_checkpoint('./model/'))
    #     w6 = load_sess.run('full_layer/weight6:0')
    #     b6 = load_sess.run('full_layer/biases6:0')
    #     w7 = load_sess.run('full_layer/weight7:0')
    #     b7 = load_sess.run('full_layer/biases7:0')
    #     w8 = load_sess.run('full_layer/weight8:0')
    #     b8 = load_sess.run('full_layer/biases8:0')
    #     with tf.variable_scope("full_layer", reuse=tf.AUTO_REUSE):
    #         weight6 = tf.get_variable(initializer=w6, name='weight6')
    #         biases6 = tf.get_variable(initializer=b6, name='biases6')
    #
    #         weight7 = tf.get_variable(initializer=w7, name='weight7')
    #         biases7 = tf.get_variable(initializer=b7, name='biases7')
    #
    #         weight8 = tf.get_variable(initializer=w8, name='weight8')
    #         biases8 = tf.get_variable(initializer=b8, name='biases8')

output6 = tf.nn.relu(tf.nn.bias_add(tf.matmul(vgg.full_input6, weight6), biases6))
output7 = tf.nn.relu(tf.nn.bias_add(tf.matmul(output6, weight7), biases7))
output = tf.nn.relu(tf.nn.bias_add(tf.matmul(output7, weight8), biases8))

prediction = tf.nn.softmax(output, name='prediction')
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=output)
                               , name='cross_entropy')  # loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learn_rate).minimize(cross_entropy)
# *****************************************************************************************************
# train net
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    print('variables initial complete, model training start.')

    # 查看结构变量
    variable_names = [v.name for v in tf.trainable_variables()]
    values = sess.run(variable_names)
    for k, v in zip(variable_names, values):
        print("Variable: ", k)
        print("Shape: ", v.shape)
    train_loss = []
    train_acc = []
    test_acc = []
    print('------------------------------------------------------------------------------------------------')
    for epoch in range(epochs):
        if epoch < int(epochs * 0.25):
            learning_rate = 0.0002
        elif int(epochs * 0.25) <= epoch < int(epochs * 0.5):
            learning_rate = 0.0001
        elif int(epochs * 0.5) <= epoch < int(epochs * 0.75):
            learning_rate = 0.00005
        elif int(epochs * 0.75) <= epoch:
            learning_rate = 0.00001
        print('epoch {0} start, learning rate : {1}.'.format(epoch + 1, learning_rate))
        time_start = time.time()

        batch_image, batch_label = load_batch_images(train_names, train_labels, image_width, batch_size)
        sess.run(optimizer, feed_dict={X: batch_image, labels: batch_label, learn_rate: learning_rate})

        train_prediction = sess.run(prediction, feed_dict={X: batch_image})
        train_accuracy = get_accuracy(train_prediction, batch_label)
        train_acc.append(train_accuracy)
        loss = sess.run(cross_entropy, feed_dict={X: batch_image, labels: batch_label})
        train_loss.append(loss)

        batch_image, batch_label = load_batch_images(test_names, test_labels, image_width, batch_size)
        test_prediction = sess.run(prediction, feed_dict={X: batch_image})
        test_accuracy = get_accuracy(test_prediction, batch_label)
        test_acc.append(test_accuracy)

        print('epoch {0} finished, train loss: {1}, train accuracy: {2}%, test accuracy: {3}%'.format(epoch + 1
                                                                                                      , loss
                                                                                                      , train_accuracy
                                                                                                      , test_accuracy))
        time_end = time.time()
        print('epoch {} time cost:'.format(epoch + 1), round((time_end - time_start) / 60, 2), 'min')
        print('------------------------------------------------------------------------------------------------')
        if epoch == 1:
            save_weights([sess.run(weight6), sess.run(biases6), sess.run(weight7)
                         , sess.run(biases7), sess.run(weight8), sess.run(biases8)])
    plot_result(train_loss, train_acc, test_acc)
    save_model(sess, 'model/vgg')
