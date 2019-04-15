import tensorflow as tf
from model.VGG16 import VGG16
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class TransforVGG16:
    def __init__(self, class_num=1000, use_layer=8, batch=32):
        self.class_num = class_num
        self.use_layer = use_layer
        self.batch_size = batch
        self.vgg = VGG16(self.class_num, training=False)

    def build(self, images):
        self.vgg.forward_propagation(images)
        if self.use_layer == 8:  # [class_num]
            self.output = self.vgg.output
        elif self.use_layer == 7:  # shape(4096)
            self.out7 = self.vgg.full_output7
            self.weight8 = self.vgg.weight(name='weight8', shape=[4096, self.class_num])
            self.biases8 = self.vgg.bias(shape=[self.class_num], name='biases8')
            self.output = tf.nn.relu(tf.nn.bias_add(tf.matmul(self.out7, self.weight8), self.biases8))
        elif self.use_layer == 6:  # [class_num]
            self.out6 = self.vgg.full_output6
            self.weight7 = self.vgg.weight(name='weight7', shape=[4096, 4096])
            self.biases7 = self.vgg.bias(shape=[4096], name='biases7')
            self.out7 = tf.nn.relu(tf.nn.bias_add(tf.matmul(self.out6, self.weight7), self.biases7))
            self.weight8 = self.vgg.weight(name='weight8', shape=[4096, self.class_num])
            self.biases8 = self.vgg.bias(shape=[self.class_num], name='biases8')
            self.output = tf.nn.relu(tf.nn.bias_add(tf.matmul(self.out7, self.weight8), self.biases8))
        elif self.use_layer == 5:  # shape(4096)
            self.input6 = tf.reshape(self.vgg.pooling5, (-1, 7 * 7 * 512))  # shape(7 * 7 * 512)
            self.weight6 = self.vgg.weight(name='weight6', shape=[7 * 7 * 512, 4096])
            self.biases6 = self.vgg.bias(shape=[4096], name='biases6')
            self.out6 = tf.nn.relu(tf.nn.bias_add(tf.matmul(self.input6, self.weight6), self.biases6))
            self.weight7 = self.vgg.weight(name='weight7', shape=[4096, 4096])
            self.biases7 = self.vgg.bias(shape=[4096], name='biases7')
            self.out7 = tf.nn.relu(tf.nn.bias_add(tf.matmul(self.out6, self.weight7), self.biases7))
            self.weight8 = self.vgg.weight(name='weight8', shape=[4096, self.class_num])
            self.biases8 = self.vgg.bias(shape=[self.class_num], name='biases8')
            self.output = tf.nn.relu(tf.nn.bias_add(tf.matmul(self.out7, self.weight8), self.biases8))
        else:
            raise ValueError('invalid "use_layer", it should be one of [5, 6, 7, 8]')
