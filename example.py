import tensorflow as tf
import numpy as np
import datetime
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

# Collect the dataset e
mnist = input_data.read_data_sets("MNIST_data/")


def discriminator(x_image, reuse=False):

    # 1st CONV + POOL LAYER
    # 32 different 5 x 5 kernels
    # First weight matrix: 5 x 5 x 32(depth)
    # Initialize to a normal distribution
    d_w1 = tf.get_variable('d_w1',
                           [5, 5, 1, 32],
                           initializer=tf.truncated_normal_initializer(mean=0, stddev=0.02))
    d_b1 = tf.get_variable('d_b1', [32], initializer=tf.constant_initializer(0))

    d1 = tf.nn.conv2d(input=x_image,
                      filter=d_w1,
                      strides=[1, 1, 1, 1],
                      padding='SAME')
    d1 = d1 + d_b1


def generator(batch_size, z_dim):
    z = tf.truncated_normal([batch_size, z_dim], mean=0, sttdev=1.0, name='z')

if __name__ == "__main__":
    sess = tf.Session()

    batch_size = 50
    z_dimensions = 100

    x_placeholder = tf.placeholder('float', shape=[None, 28, 28, 1], name='placeholder')

    Gz = generator(batch_size, z_dimensions)
