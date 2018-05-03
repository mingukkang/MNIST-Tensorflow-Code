import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import zipfile
from glob import *

#initializer = tf.contrib.layers.xavier_initializer()
initializer = tf.contrib.layers.variance_scaling_initializer(factor = 1.0)

def zip_extractor(folder):
    now = os.getcwd()
    zip_list = glob(folder + "/*.zip")
    for i in range(len(zip_list)):
        s = os.path.splitext(zip_list[i])[0]

        if not os.path.exists(s):
            os.makedirs(s)
            fzip = zipfile.ZipFile(zip_list[i], 'r')
            fzip.extractall(path = s + "/")
            fzip.close()

def conv(inputs,filters,name):
    net = tf.layers.conv2d(inputs = inputs,
                           filters = filters,
                           kernel_size = [3,3],
                           strides = (1,1),
                           padding ="SAME",
                           kernel_initializer = initializer,
                           name = name,
                           reuse = tf.AUTO_REUSE)
    return net

def maxpool(input,name):
    net = tf.nn.max_pool(value = input, ksize = [1,2,2,1], strides = [1,2,2,1], padding = "SAME", name = name)
    return net

def strided_conv(inputs,name):
    net = tf.layers.conv2d(inputs = inputs,
                           filters = inputs.get_shape()[3],
                           kernel_size = [3,3],
                           strides = (2,2),
                           padding = "SAME",
                           kernel_initializer = initializer,
                           name = name,
                           reuse = tf.AUTO_REUSE)
    return net

def bn(inputs,is_training,name):
    net = tf.contrib.layers.batch_norm(inputs, decay = 0.9, is_training = is_training, reuse = tf.AUTO_REUSE, scope = name)
    return net

def leaky(input):
    return tf.nn.leaky_relu(input)

def dense(inputs, units, name):
    net = tf.layers.dense(inputs = inputs,
                          units = units,
                          reuse = tf.AUTO_REUSE,
                          name = name,
                          kernel_initializer = initializer)
    return net

def get_Accuracy(hypothesis, labels):
    with tf.name_scope("Accuracy"):
        # labels should be a one hot encoded value.
        # hypothesis is the value after softmax activation.
        is_correct = np.equal(np.argmax(hypothesis,axis =1), np.argmax(labels, axis =1))
        accuracy = np.mean(is_correct.astype(dtype = np.float32))*100
    return accuracy

def plot_data_label(images,labels,channels,width,height,figsize):
    images = np.multiply(images,255)
    load_size = images.shape[1]
    labels = np.argmax(labels,axis =1)
    fig, axes = plt.subplots(width,height,figsize = (figsize,figsize))
    fig.subplots_adjust(hspace=1, wspace=1)
    path = os.getcwd() + "/plt_images"
    if not os.path.exists(path):
        os.makedirs(path)
    for i, ax in enumerate(axes.flat):
        if channels == 1:
            ax.imshow(images[i].reshape(load_size, load_size))
        else:
            ax.imshow(images[i].reshape(load_size, load_size, channels))
        ax.set_xlabel("label: %d" % (labels[i]))
        file_name = path + "/image" + str(i)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.savefig(file_name)
    plt.close()