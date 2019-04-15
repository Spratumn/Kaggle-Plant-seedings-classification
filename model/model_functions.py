import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt


def get_accuracy(predictions, labels):
    p = np.argmax(predictions, axis=1)
    o = np.argmax(labels, axis=1)
    crrect = np.equal(p, o)
    accuracy = 100. * ((crrect > 0).sum()) / len(crrect)
    return accuracy


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
def save_model(sess, path):
    saver = tf.train.Saver()
    saver.save(sess, path)
    print('Trained Model Saved.')


def get_prediction_label(prediction, file_path, hastop5=False):
    label_list = np.array([l.strip() for l in open(file_path).readlines()])
    prediction_index = np.argsort(prediction)[::-1]
    label_top1 = label_list[prediction_index[0]]
    label_top5 = label_top1
    if hastop5:
        label_top5 = label_list[prediction_index[0:5]]
    return label_top1, label_top5

