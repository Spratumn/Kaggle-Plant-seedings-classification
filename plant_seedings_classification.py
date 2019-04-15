import os
import time
from model.model_functions import *
from model.TransferVGG16 import TransforVGG16
from load_image_by_name import get_name_label_set, load_batch_images
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


if __name__ == '__main__':
    image_width = 224
    batch_size = 64
    epochs = 200
    class_num = 12
    class_list = ['Black-grass', 'Charlock', 'Cleavers', 'Common Chickweed'
                  , 'Common wheat', 'Fat Hen', 'Loose Silky-bent', 'Maize'
                  , 'Scentless Mayweed', 'Shepherds Purse', 'Small-flowered Cranesbill', 'Sugar beet']
    train_names, train_labels, test_names, test_labels = get_name_label_set('./dataset/train', class_list)

    # 输入样本
    X = tf.placeholder(tf.float32, shape=[batch_size, image_width, image_width, 3], name='X-input')
    # 真实标签
    labels = tf.placeholder(tf.float32, shape=[batch_size, class_num], name='labels-input')
    learn_rate = tf.placeholder(tf.float16, name='learn_rate')

    net_build_start = time.time()
    transfer = TransforVGG16(12, use_layer=7, batch=batch_size)
    transfer.build(X)
    net_build_end = time.time()
    print('build net finished {} s'.format(net_build_end - net_build_start))

    prediction = tf.nn.softmax(transfer.output, name='prediction')
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=transfer.output)
                                   , name='cross_entropy')  # loss
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learn_rate).minimize(cross_entropy)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        print('variables initial complete, model training start.')

        # 查看结构变量
        # variable_names = [v.name for v in tf.trainable_variables()]
        # values = sess.run(variable_names)
        # for k, v in zip(variable_names, values):
        #     print("Variable: ", k)
        #     print("Shape: ", v.shape)
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
            # w = sess.run(transfer.weight8[0][0:5])
            # b = sess.run(transfer.biases8[0:5])
            # print(w)
            # print(b)
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

        plot_result(train_loss, train_acc, test_acc)
        save_model(sess, './model/my_trained_Vgg16')


