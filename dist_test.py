from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import math
import seaborn as sns

def points_on_circle(num, r):
  pi = 3.141592
  coords = [ ( r * math.cos((2. / num) * i * pi)  , r * math.sin((2. / num) * i * pi) ) for i in range(num)]
  return coords

def mdist():
  p = points_on_circle(8, 2.)
  dist = tf.contrib.distributions # pylint: disable=E1101
  gauss = dist.Mixture(
  cat=dist.Categorical(probs=[0.25 for _ in range(8)]),
  components=[    
    dist.MultivariateNormalDiag(loc=p[i], scale_diag=[0.02, 0.02]) for i in range(8)
  ])
  return gauss

def draw2():
  gauss = mdist()
  samples = gauss.sample([15000])

  with tf.Session() as sess:
    samples_res = sess.run(samples)
    
  sns.color_palette('Oranges', n_colors=256)[0]
  sns.kdeplot(samples_res[:, 0], samples_res[:, 1], shade=True, cmap='Oranges', n_levels=20, clip=[[-4,4]]*2)
  plt.show()

def draw1():
  gauss = mdist()
  
  X, Y = tf.meshgrid(tf.range(-3, 3, 0.01), tf.range(-3, 3, 0.01))
  idx = tf.concat([tf.reshape(X, [-1, 1]), tf.reshape(Y,[-1,1])], axis =1)
  prob = gauss.prob(idx)
  prob = tf.reshape(prob, tf.shape(X))

  with tf.Session() as sess:
    prob_res = sess.run(prob)

  plt.imshow(prob_res, cmap='terrain', interpolation='nearest', origin='lower', extent=[-2, 2, -2, 2])
  plt.show()

draw2()