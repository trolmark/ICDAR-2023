import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import math


def rand_contrast(image, heatmap):
  probability = 0.5
  min_factor = 0.1
  max_factor = 0.3

  r = tf.random.uniform(shape=[], dtype=tf.dtypes.float32)
  t = tf.random.uniform([], min_factor, max_factor, tf.dtypes.float32)
  result = tf.cond(r > probability, lambda: (image, heatmap),
                    lambda: (tf.image.adjust_brightness(image, t), heatmap))
  return result

def rand_distort(image, heatmap):
  probability = 0.5
  max_angle = 0.3

  r = tf.random.uniform(shape=[], dtype=tf.dtypes.float32)
  theta = tf.random.uniform([], -max_angle, max_angle, tf.dtypes.float32)
  transform = [1, tf.sin(theta), 0, 0, tf.cos(theta), 0, 0, 0]

  result = tf.cond(r > probability,
    lambda: (image, heatmap),
    lambda: (tfa.image.transform(image, transform, fill_value = 255.0),
            tfa.image.transform(heatmap, transform, fill_value = 0.0)))
  return result


def rand_rotate(image, heatmap):
  probability = 0.5
  max_angle = 75.0
  max_angle = max_angle * math.pi / 180.0

  r = tf.random.uniform(shape=[], dtype=tf.dtypes.float32)
  theta = tf.random.uniform([], -max_angle, max_angle, tf.dtypes.float32)
  result = tf.cond(r > probability,
    lambda: (image, heatmap),
    lambda: (tfa.image.rotate(image, theta, fill_value = 255.0),
            tfa.image.rotate(heatmap, theta, fill_value = 0.0)))

  return result


