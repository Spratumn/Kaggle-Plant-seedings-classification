import tensorflow as tf
import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class VGG16:
    def __init__(self, num_class=1000, training=True):
        self.num_class = num_class
        if training:
            self.weight1_1 = self.weight(name='weight1_1', shape=[3, 3, 3, 64])
            self.biases1_1 = self.bias(shape=[64], name='biases1_1')
            self.weight1_2 = self.weight(name='weight1_2', shape=[3, 3, 64, 64])
            self.biases1_2 = self.bias(shape=[64], name='biases1_2')

            self.weight2_1 = self.weight(name='weight2_1', shape=[3, 3, 64, 128])
            self.biases2_1 = self.bias(shape=[128], name='biases2_1')
            self.weight2_2 = self.weight(name='weight2_2', shape=[3, 3, 128, 128])
            self.biases2_2 = self.bias(shape=[128], name='biases2_2')

            self.weight3_1 = self.weight(name='weight3_1', shape=[3, 3, 128, 256])
            self.biases3_1 = self.bias(shape=[256], name='biases3_1')
            self.weight3_2 = self.weight(name='weight3_2', shape=[3, 3, 256, 256])
            self.biases3_2 = self.bias(shape=[256], name='biases3_2')
            self.weight3_3 = self.weight(name='weight3_3', shape=[3, 3, 256, 256])
            self.biases3_3 = self.bias(shape=[256], name='biases3_3')

            self.weight4_1 = self.weight(name='weight4_1', shape=[3, 3, 256, 512])
            self.biases4_1 = self.bias(shape=[512], name='biases4_1')
            self.weight4_2 = self.weight(name='weight4_2', shape=[3, 3, 512, 512])
            self.biases4_2 = self.bias(shape=[512], name='biases4_2')
            self.weight4_3 = self.weight(name='weight4_3', shape=[3, 3, 512, 512])
            self.biases4_3 = self.bias(shape=[512], name='biases4_3')

            self.weight5_1 = self.weight(name='weight5_1', shape=[3, 3, 512, 512])
            self.biases5_1 = self.bias(shape=[512], name='biases5_1')
            self.weight5_2 = self.weight(name='weight5_2', shape=[3, 3, 512, 512])
            self.biases5_2 = self.bias(shape=[512], name='biases5_2')
            self.weight5_3 = self.weight(name='weight5_3', shape=[3, 3, 512, 512])
            self.biases5_3 = self.bias(shape=[512], name='biases5_3')

            self.weight6 = self.weight(name='weight6', shape=[7 * 7 * 512, 4096])
            self.biases6 = self.bias(shape=[4096], name='biases6')

            self.weight7 = self.weight(name='weight7', shape=[4096, 4096])
            self.biases7 = self.bias(shape=[4096], name='biases7')

            self.weight8 = self.weight(name='weight8', shape=[4096, self.num_class])
            self.biases8 = self.bias(shape=[self.num_class], name='biases8')
        else:
            self.data_dict = np.load('model/vgg16.npy', encoding='latin1').item()
            self.weight1_1 = tf.constant(self.data_dict['conv1_1'][0], name="conv1_1_weights")
            self.biases1_1 = tf.constant(self.data_dict['conv1_1'][1], name="conv1_1_biases")
            self.weight1_2 = tf.constant(self.data_dict['conv1_2'][0], name="conv1_2_weights")
            self.biases1_2 = tf.constant(self.data_dict['conv1_2'][1], name="conv1_2_biases")

            self.weight2_1 = tf.constant(self.data_dict['conv2_1'][0], name="conv2_1_weights")
            self.biases2_1 = tf.constant(self.data_dict['conv2_1'][1], name="conv2_1_biases")
            self.weight2_2 = tf.constant(self.data_dict['conv2_2'][0], name="conv2_2_weights")
            self.biases2_2 = tf.constant(self.data_dict['conv2_2'][1], name="conv2_2_biases")

            self.weight3_1 = tf.constant(self.data_dict['conv3_1'][0], name="conv3_1_weights")
            self.biases3_1 = tf.constant(self.data_dict['conv3_1'][1], name="conv3_1_biases")
            self.weight3_2 = tf.constant(self.data_dict['conv3_2'][0], name="conv3_2_weights")
            self.biases3_2 = tf.constant(self.data_dict['conv3_2'][1], name="conv3_2_biases")
            self.weight3_3 = tf.constant(self.data_dict['conv3_3'][0], name="conv3_3_weights")
            self.biases3_3 = tf.constant(self.data_dict['conv3_3'][1], name="conv3_3_biases")

            self.weight4_1 = tf.constant(self.data_dict['conv4_1'][0], name="conv4_1_weights")
            self.biases4_1 = tf.constant(self.data_dict['conv4_1'][1], name="conv4_1_biases")
            self.weight4_2 = tf.constant(self.data_dict['conv4_2'][0], name="conv4_2_weights")
            self.biases4_2 = tf.constant(self.data_dict['conv4_2'][1], name="conv4_2_biases")
            self.weight4_3 = tf.constant(self.data_dict['conv4_3'][0], name="conv4_3_weights")
            self.biases4_3 = tf.constant(self.data_dict['conv4_3'][1], name="conv4_3_biases")

            self.weight5_1 = tf.constant(self.data_dict['conv5_1'][0], name="conv5_1_weights")
            self.biases5_1 = tf.constant(self.data_dict['conv5_1'][1], name="conv5_1_biases")
            self.weight5_2 = tf.constant(self.data_dict['conv5_2'][0], name="conv5_2_weights")
            self.biases5_2 = tf.constant(self.data_dict['conv5_2'][1], name="conv5_2_biases")
            self.weight5_3 = tf.constant(self.data_dict['conv5_3'][0], name="conv5_3_weights")
            self.biases5_3 = tf.constant(self.data_dict['conv5_3'][1], name="conv5_3_biases")

            self.weight6 = tf.constant(self.data_dict['fc6'][0], name="fc6_weights")
            self.biases6 = tf.constant(self.data_dict['fc6'][1], name="fc6_biases")

            self.weight7 = tf.constant(self.data_dict['fc7'][0], name="fc7_weights")
            self.biases7 = tf.constant(self.data_dict['fc7'][1], name="fc7_biases")

            self.weight8 = tf.constant(self.data_dict['fc8'][0], name="fc8_weights")
            self.biases8 = tf.constant(self.data_dict['fc8'][1], name="fc8_biases")

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
        return tf.get_variable(initializer=initial, name=name, shape=shape, dtype=dtype)

    def bias(self, name, shape, dtype=tf.float32):
        initial = tf.constant_initializer(0.0)
        return tf.get_variable(initializer=initial, name=name, shape=shape, dtype=dtype)

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

