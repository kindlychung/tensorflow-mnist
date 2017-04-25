import tensorflow as tf
import mnist.model as model

import os
import mnist.model as model
import tensorflow as tf


checkpoint_file = os.path.join(os.path.dirname(__file__), 'data', 'convolutional1.ckpt')
mnist_data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'MNIST_data')

# checkpoint_file = "/home/kaiyin/PycharmProjects/tensorflow-mnist/mnist/data/convolutional1.ckpt"
# mnist_data_dir = "/home/kaiyin/PycharmProjects/tensorflow-mnist/MNIST_data"
from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets(mnist_data_dir, one_hot=True)

from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
from mnist.util.plotting import show_flat_image
img = data.train.images[0]
show_flat_image(img, 28, 28)



x = tf.placeholder("float", [None, 784])
sess = tf.Session()

convnet_checkpoint = "mnist/data/convolutional1.ckpt"
with tf.variable_scope("convolutional"):
    keep_prob = tf.placeholder("float")
    y2, variables = model.convolutional(x, keep_prob)
saver = tf.train.Saver(variables)
saver.restore(sess, convnet_checkpoint)