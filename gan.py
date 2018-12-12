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

    # Convolution layer: 28 x 28 x 3 ---> 22 x 22 x 32
    d1 = tf.nn.conv2d(input=x_image,
                      filter=d_w1,
                      strides=[1, 1, 1, 1],
                      padding='SAME')
    d1 = d1 + d_b1
    d1 = tf.nn.relu(d1)
    d1 = tf.nn.avg_pool(d1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # 2ND CONV + POOL LAYER
    # 64 5 x 5 filters
    d_w2 = tf.get_variable('d_w2',
                           [5, 5, 1, 64],
                           initializer=tf.truncated_normal_initializer(stddev=0.002))
    d_b2 = tf.get_variable('d_b2', [64], initializer=tf.constant_initializer(0))

    d2 = tf.nn.conv2d(input=d1,
                      filter=d_w2,
                      strides=[1, 1, 1, 1],
                      padding='SAME')
    d2 = d2 + d_b2
    d2 = tf.nn.relu(d2)
    d2 = tf.nn.avg_pool(d2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # 3RD LAYER - FULLY CONNECTED
    # 7 x 7 x 64 --> 1024
    d_w3 = tf.get_variable('d_w3',
                           [7 * 7 * 64, 1024],
                           initializer=tf.truncated_normal_initializer(stddev=0.002))
    d_b3 = tf.get_variable('d_b3',
                           [1024],
                           initializer=tf.constant_initializer(0))

    d3 = tf.reshape(d2, [-1, 7 * 7 * 64])
    d3 = tf.matmul(d3, d_w3)
    d3 = d3 + d_b3

    d3 = tf.nn.relu(d3)

    # Final layer - fully connected
    # 1024 -> 1
    d_w4 = tf.get_variable('d_w4', [1024, 1], initializer=tf.truncated_normal_initializer(stddev=0.02))
    d_b4 = tf.get_variable('d_b4', [1], initializer=tf.constant_initializer(0))

    # Final layer
    d4 = tf.matmul(d3, d_w4) + d_b4
    # d4 dimensions: batch_size x 1

    return d4

def generator(batch_size, z_dim):
    z = tf.truncated_normal([batch_size, z_dim], mean=0, sttdev=1.0, name='z')

if __name__ == "__main__":
    sess = tf.Session()

    batch_size = 50
    z_dimensions = 100

    x_placeholder = tf.placeholder('float', shape=[None, 28, 28, 1], name='placeholder')

    Gz = generator(batch_size, z_dimensions)
