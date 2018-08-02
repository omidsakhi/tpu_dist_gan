from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import math

class InputFunction(object):  
  
  def __init__(self, noise_dim):    
    self.noise_dim = noise_dim

  def points_on_circle(self, num, r):
    pi = 3.141592
    coords = [ (r * math.cos((2. / num) * i * pi)  , r * math.sin((2. / num) * i * pi) ) for i in range(num)]
    return coords


  def __call__(self, params):      
    batch_size = params['batch_size']    
    random_noise = tf.random_normal([batch_size, self.noise_dim])
    dist = tf.contrib.distributions # pylint: disable=E1101
    p = self.points_on_circle(8, 2.)
    gauss = dist.Mixture(
    cat=dist.Categorical(probs=[0.25 for _ in range(8)]),    
    components=[
        dist.MultivariateNormalDiag(loc=p[i], scale_diag=[0.02, 0.02]) for i in range(8)
    ])
    samples = gauss.sample([batch_size])
    features = {
        'samples': samples,
        'random_noise': random_noise}

    return features
