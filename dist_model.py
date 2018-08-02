from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def _leaky_relu(x):
  return tf.nn.leaky_relu(x, alpha=0.2)


def _batch_norm(x, is_training, name):
  return tf.layers.batch_normalization(
      x, momentum=0.0001, epsilon=1e-5, training=is_training, fused=True, name=name)


def _dense(x, channels, name):
  return tf.layers.dense(
      x, channels,
      bias_initializer=None,
      use_bias=False,
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
      name=name)

def _dense_with_bias(x, channels, name):
  return tf.layers.dense(
      x, channels,
      bias_initializer=tf.zeros_initializer(),      
      use_bias=True,
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
      name=name)

def _conv2d(x, filters, kernel_size, stride, name):
  return tf.layers.conv2d(
      x, filters, [kernel_size, kernel_size],
      strides=[stride, stride], padding='same',
      bias_initializer=None,
      use_bias=False,
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
      name=name)

def _conv2d_with_bias(x, filters, kernel_size, stride, name):
  return tf.layers.conv2d(
      x, filters, [kernel_size, kernel_size],
      strides=[stride, stride], padding='same',
      bias_initializer=tf.zeros_initializer(),      
      use_bias=True,
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
      name=name)

def _deconv2d(x, filters, kernel_size, stride, name):
  return tf.layers.conv2d_transpose(
      x, filters, [kernel_size, kernel_size],
      strides=[stride, stride], padding='same',
      bias_initializer=None,
      use_bias=False,
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
      name=name)

def _deconv2d_with_bias(x, filters, kernel_size, stride, name):
  return tf.layers.conv2d_transpose(
      x, filters, [kernel_size, kernel_size],
      strides=[stride, stride], padding='same',
      bias_initializer=tf.zeros_initializer(),      
      use_bias=True,
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
      name=name)

def _dblock(x, is_training, name, with_bn=False):
    x = _dense(x, 32, name + "_fc")
    #x = tf.nn.dropout(x, keep_prob=0.75 if is_training else 1, name = name + "_do")
    if with_bn:
      x = _batch_norm(x, is_training, name + "_bn")   
      #x = _op(x) 
    x = _leaky_relu(x)
    #x = tf.nn.relu(x)
    return x

def discriminator(x, is_training=True, scope='Discriminator'):
  with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):

    x = _dblock(x, is_training, "d1")
    x = _dblock(x, is_training, "d2", with_bn=True)
    #x = _dblock(x, is_training, "d3", with_bn=True)
    x = _dense_with_bias(x, 1, 'd4')    
    
    return x

def _gblock(x, is_training, name):
    x = _dense(x, 128, name + "_fc")
    x = _batch_norm(x, is_training, name + "_bn")    
    #x = tf.nn.dropout(x, keep_prob=0.75 if is_training else 1, name = name + "_do")        
    #x = _leaky_relu(x)
    x = tf.nn.relu(x)
    return x

def generator(x, is_training=True, scope='Generator'):
  with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        
    x = _gblock(x, is_training, "g1")
    x = _gblock(x, is_training, "g2")
    #x = _gblock(x, is_training, "g3")

    x = _dense_with_bias(x, 2, 'g4')

    return x