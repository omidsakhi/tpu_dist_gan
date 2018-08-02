from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

# Standard Imports
from absl import flags
import absl.logging as _logging  # pylint: disable=unused-import
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import dist_input
import dist_model
from tensorflow.contrib.tpu.python.tpu import tpu_config
from tensorflow.contrib.tpu.python.tpu import tpu_estimator
from tensorflow.contrib.tpu.python.tpu import tpu_optimizer
from tensorflow.python.estimator import estimator # pylint: disable=E0611
import seaborn as sns

FLAGS = flags.FLAGS

# Cloud TPU Cluster Resolvers
flags.DEFINE_string(
    'tpu', default=None,
    help='The Cloud TPU to use for training. This should be either the name '
    'used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 url.')
flags.DEFINE_string(
    'gcp_project', default=None,
    help='Project name for the Cloud TPU-enabled project. If not specified, we '
    'will attempt to automatically detect the GCE project from metadata.')
flags.DEFINE_string(
    'tpu_zone', default=None,
    help='GCE zone where the Cloud TPU is located in. If not specified, we '
    'will attempt to automatically detect the GCE project from metadata.')

flags.DEFINE_string('data_dir', './dataset', 'Bucket/Folder that contains the data tfrecord files')
flags.DEFINE_string('model_dir', './output', 'Output model directory')
flags.DEFINE_integer('noise_dim', 256,
                     'Number of dimensions for the noise vector')
flags.DEFINE_integer('batch_size', 128,
                     'Batch size for both generator and discriminator')
flags.DEFINE_integer('num_shards', None, 'Number of TPU chips')
flags.DEFINE_integer('train_steps', 500000, 'Number of training steps')
flags.DEFINE_integer('train_steps_per_eval', 1000,
                     'Steps per eval and image generation')
flags.DEFINE_integer('iterations_per_loop', 100,
                     'Steps per interior TPU loop. Should be less than'
                     ' --train_steps_per_eval')
flags.DEFINE_float('learning_rate', 0.002, 'Learning rate for both D and G')
flags.DEFINE_float('beta1', 0.5, 'Beta1 for both D and G')
flags.DEFINE_boolean('eval_loss', False,
                     'Evaluate discriminator and generator loss during eval')
flags.DEFINE_integer('num_eval_samples', 2048,
                     'Number of samples for evaluate')
flags.DEFINE_boolean('use_tpu', False, 'Use TPU for training')

# Global variables for data and model
dataset = None
model = None

def WGAN(real, fake):
    real = tf.sigmoid(real)
    fake = tf.sigmoid(fake)
    return tf.reduce_mean(real) - tf.reduce_mean(fake), tf.reduce_mean(fake)

def LSGAN(real, fake):
    return tf.square(real - 1) + tf.square(fake), tf.square(fake - 1)

def GAN(real, fake):

    d_loss_on_data = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.ones_like(real),
        logits=real)
    d_loss_on_gen = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.zeros_like(fake),
        logits=fake)

    d_loss = d_loss_on_data + d_loss_on_gen
    
    g_loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.ones_like(fake),
        logits=fake)

    return d_loss, g_loss

def model_fn(features, labels, mode, params):
  """Constructs DCGAN from individual generator and discriminator networks."""
  del labels    # Unconditional GAN does not use labels

  if mode == tf.estimator.ModeKeys.PREDICT:
    ########### 
    # PREDICT #
    ###########
    # Pass only noise to PREDICT mode
    random_noise = features['random_noise']    
    predictions = {
        'generated_samples': model.generator(random_noise, is_training=False)        
    }

    return tpu_estimator.TPUEstimatorSpec(mode=mode, predictions=predictions)
  
  batch_size = params['batch_size']   # pylint: disable=unused-variable
  real_samples = features['samples']
  random_noise = features['random_noise']

  is_training = (mode == tf.estimator.ModeKeys.TRAIN)
  generated_samples = model.generator(random_noise, is_training=is_training)

  # Get logits from discriminator
  d_real = model.discriminator(real_samples)
  d_fake = model.discriminator(generated_samples)  
    
  d_loss, g_loss = LSGAN(d_real,d_fake)

  if mode == tf.estimator.ModeKeys.TRAIN:
    #########
    # TRAIN #
    #########    
    d_loss = tf.reduce_mean(d_loss)
    g_loss = tf.reduce_mean(g_loss)
    
    d_optimizer = tf.train.AdamOptimizer(
        learning_rate=FLAGS.learning_rate, beta1=FLAGS.beta1)
    g_optimizer = tf.train.AdamOptimizer(
        learning_rate=FLAGS.learning_rate, beta1=FLAGS.beta1)

    if FLAGS.use_tpu:
      d_optimizer = tpu_optimizer.CrossShardOptimizer(d_optimizer)
      g_optimizer = tpu_optimizer.CrossShardOptimizer(g_optimizer)      

    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
      d_step = d_optimizer.minimize(
          d_loss,
          var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                     scope='Discriminator'))
      g_step = g_optimizer.minimize(
          g_loss,
          var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                     scope='Generator'))

      increment_step = tf.assign_add(tf.train.get_or_create_global_step(), 1)
      joint_op = tf.group([d_step, g_step, increment_step])

      return tpu_estimator.TPUEstimatorSpec(
          mode=mode,
          loss=g_loss,
          train_op=joint_op)

  elif mode == tf.estimator.ModeKeys.EVAL:
    ########
    # EVAL #
    ########
    def _eval_metric_fn(d_loss, g_loss):
      # When using TPUs, this function is run on a different machine than the
      # rest of the model_fn and should not capture any Tensors defined there
      return {
          'discriminator_loss': tf.metrics.mean(d_loss),
          'generator_loss': tf.metrics.mean(g_loss)}

    return tpu_estimator.TPUEstimatorSpec(
        mode=mode,
        loss=tf.reduce_mean(g_loss),
        eval_metrics=(_eval_metric_fn, [d_loss, g_loss]))

  # Should never reach here
  raise ValueError('Invalid mode provided to model_fn')


def generate_input_fn(is_training):
  """Creates input_fn depending on whether the code is training or not."""
  return dataset.InputFunction(FLAGS.noise_dim)


def noise_input_fn(params):
  """Input function for generating samples for PREDICT mode.
  Generates a single Tensor of fixed random noise. Use tf.data.Dataset to
  signal to the estimator when to terminate the generator returned by
  predict().
  Args:
    params: param `dict` passed by TPUEstimator.
  Returns:
    1-element `dict` containing the randomly generated noise.
  """
  np.random.seed(0)
  noise_dataset = tf.data.Dataset.from_tensors(tf.constant(
      np.random.randn(params['batch_size'], FLAGS.noise_dim), dtype=tf.float32))
  noise = noise_dataset.make_one_shot_iterator().get_next()
  return {'random_noise': noise}, None


def main(argv):

  del argv
  
  tpu_cluster_resolver = None
  
  if FLAGS.use_tpu:
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver( # pylint: disable=E1101
      FLAGS.tpu,
      zone=FLAGS.tpu_zone,
      project=FLAGS.gcp_project)

  config = tpu_config.RunConfig(
      cluster=tpu_cluster_resolver,
      model_dir=FLAGS.model_dir,
      tpu_config=tpu_config.TPUConfig(
          num_shards=FLAGS.num_shards,
          iterations_per_loop=FLAGS.iterations_per_loop))

  # Set module-level global variable so that model_fn and input_fn can be
  # identical for each different kind of dataset and model
  global dataset, model  
  dataset = dist_input
  model = dist_model

  # TPU-based estimator used for TRAIN and EVAL
  est = tpu_estimator.TPUEstimator(
      model_fn=model_fn,
      use_tpu=FLAGS.use_tpu,
      config=config,
      params={"data_dir": FLAGS.data_dir},
      train_batch_size=FLAGS.batch_size,
      eval_batch_size=FLAGS.batch_size)

  # CPU-based estimator used for PREDICT (generating images)
  cpu_est = tpu_estimator.TPUEstimator(
      model_fn=model_fn,
      use_tpu=False,
      config=config,
      params={"data_dir": FLAGS.data_dir},
      predict_batch_size=FLAGS.num_eval_samples)

  tf.gfile.MakeDirs(os.path.join(FLAGS.model_dir))
  tf.gfile.MakeDirs(os.path.join(FLAGS.model_dir, 'generated_samples'))

  current_step = estimator._load_global_step_from_checkpoint_dir(FLAGS.model_dir)   # pylint: disable=protected-access,line-too-long
  tf.logging.info('Starting training for %d steps, current step: %d' %
                  (FLAGS.train_steps, current_step))
  while current_step < FLAGS.train_steps:
    next_checkpoint = min(current_step + FLAGS.train_steps_per_eval,
                          FLAGS.train_steps)
    est.train(input_fn=generate_input_fn(True),
              max_steps=next_checkpoint)
    current_step = next_checkpoint
    tf.logging.info('Finished training step %d' % current_step)

    if FLAGS.eval_loss:
      # Evaluate loss on test set
      metrics = est.evaluate(input_fn=generate_input_fn(False),
                             steps=FLAGS.num_eval_samples // FLAGS.batch_size)
      tf.logging.info('Finished evaluating')
      tf.logging.info(metrics)

    # Render some generated samples
    generated_iter = cpu_est.predict(input_fn=noise_input_fn)
    generated_samples = [p['generated_samples'][:] for p in generated_iter]    
    assert len(generated_samples) == FLAGS.num_eval_samples
    generated_samples = np.vstack(generated_samples)    
    step_string = str(current_step).zfill(5)
    file_name = FLAGS.model_dir + "/generated_samples/" + 'gen_%s.png' % (step_string)
    plt.clf()    

    #plt.plot(generated_samples[:,0], generated_samples[:,1], 'b.')    
    #axes = plt.gca()
    #axes.set_xlim([-4,4])
    #axes.set_ylim([-4,4])    

    sns.color_palette('Oranges', n_colors=256)[0]
    sns.kdeplot(generated_samples[:, 0], generated_samples[:, 1], shade=True, cmap='Oranges', n_levels=20, clip=[[-4,4]]*2)    
    
    plt.title('Step: {}'.format(current_step))
    plt.savefig(file_name, bbox_inches='tight')
    tf.logging.info('Finished generating figure')


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)  
  tf.app.run(main)