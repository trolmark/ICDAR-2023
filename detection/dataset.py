import tensorflow as tf
import constants as cnt
from heatmap import CornersHeatmap
import numpy as np
from augmentation import (
  rand_contrast, 
  rand_distort,
  rand_rotate
)

def parse_tfrecord_fn(example):    
    feature_description = {
      'height': tf.io.FixedLenFeature([], tf.int64),
      'width': tf.io.FixedLenFeature([], tf.int64),
      'depth': tf.io.FixedLenFeature([], tf.int64),
      'raw_image' : tf.io.FixedLenFeature([], tf.string),
      'image/keypoints/x': tf.io.VarLenFeature(dtype = tf.float32),
      'image/keypoints/y': tf.io.VarLenFeature(dtype = tf.float32),
    }
    example = tf.io.parse_single_example(example, feature_description)
    return example

def load_image_with_keypoints(example):
    height = example['height']
    width = example['width']
    depth = example['depth']
    raw_image = example['raw_image']
    pxls = tf.io.parse_tensor(raw_image, out_type=tf.uint8)
    pxls = tf.reshape(pxls, shape=[height, width,depth])
    pxls = tf.image.resize(pxls, (cnt.INPUT_SIZE, cnt.INPUT_SIZE))

    xs = tf.cast(tf.expand_dims(example['image/keypoints/x'].values, 0), dtype=tf.float32) 
    ys = tf.cast(tf.expand_dims(example['image/keypoints/y'].values, 0), dtype=tf.float32)
   
    keypoints = tf.cast(tf.stack([xs, ys]), dtype = tf.float32)      
    keypoints = tf.concat(axis=0, values=[keypoints[0], keypoints[1]]) 
    keypoints = tf.transpose(keypoints, [1, 0])

    h, w = tf.cast(height, dtype = tf.float32), tf.cast(width, dtype = tf.float32)
    keypoints /= tf.stack([w, h])
    keypoints *= cnt.INPUT_SIZE
    keypoints = tf.cast(keypoints, dtype = tf.int64) 
    
    padding_add = cnt.MAX_KEYPOINTS_ON_IMAGE - tf.shape(keypoints)[0]
    keypoints = tf.pad(keypoints, tf.stack([[0, padding_add], [0, 0]]))
        
    return pxls, keypoints

def prepare_heatmap(pxls, keypoints):
  image_shape = tf.shape(pxls)
  height, width = image_shape[0], image_shape[1]

  heatmap = CornersHeatmap.tf_heatmap(
      keypoints,
      cnt.HEATMAP_KERNEL_SIGMA,
      cnt.MAX_KEYPOINTS_ON_IMAGE,
      im_h=height,
      im_w=width
  )

  output_stride = cnt.OUTPUT_STRIDE
  output_size = (
    tf.cast(width / output_stride, dtype = tf.int32), 
    tf.cast(height / output_stride, dtype = tf.int32)
  )
        
  heatmap = tf.expand_dims(tf.reduce_sum(heatmap,axis = 2), axis = 2)
  heatmap = tf.image.resize(heatmap, output_size)
  return pxls, heatmap


def tfrecord_ds(path_to_dataset, training: bool = True):
  if not isinstance(path_to_dataset, type(list)):
    path_to_dataset = [path_to_dataset]
  
  dataset = tf.data.TFRecordDataset(path_to_dataset)

  dataset = (dataset
    .map(parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)
    .map(load_image_with_keypoints, num_parallel_calls=tf.data.AUTOTUNE)
    .map(prepare_heatmap, num_parallel_calls=tf.data.AUTOTUNE)
    .cache()
  )
  
  if training:
    dataset = (dataset
      .map(rand_contrast)
      .map(rand_distort)
      .map(rand_rotate)
    )
    dataset = dataset.shuffle(2 * cnt.BATCH_SIZE)
  
  dataset = (dataset
    .batch(cnt.BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
  )
  
  return dataset
