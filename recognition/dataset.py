import tensorflow as tf
import constants as cnt
import numpy as np


class DatasetLoader(object):

  def __init__(self, out_charset, start_char, end_char, max_length = 32):
    self.out_charset = out_charset
    
    vocabulary = out_charset + [start_char, end_char]
    self.char_to_num = tf.keras.layers.StringLookup(
      vocabulary=list(vocabulary), mask_token=None
    )
    self.num_to_char = tf.keras.layers.StringLookup(
      vocabulary=self.char_to_num.get_vocabulary(), mask_token=None, invert=True
    )

    self.start_char = start_char
    self.end_char = end_char

    mask_idxs = [0, 1] # For [PAD] and [UNK] tokens

    if start_char != '' and end_char != '': 
      self.start_token = self.char_to_num(start_char)
      self.end_token = self.char_to_num(end_char)
      max_length += 2 # For [START] and [END] tokens
      mask_idxs.append(self.start_token)

    # Prevent from generating padding, unknown, or start when using argmax in model.predict
    token_mask = np.zeros([self.char_to_num.vocabulary_size()], dtype=bool)
    token_mask[np.array(mask_idxs)] = True
    self.token_mask = token_mask
    self.max_length = max_length

  def _parse_tfrecord_fn(self, example):    
      feature_description = {
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'depth': tf.io.FixedLenFeature([], tf.int64),
        'masked_image' : tf.io.FixedLenFeature([], tf.string),
        'text/string': tf.io.FixedLenFeature([], dtype=tf.string, default_value=''),
        'text/length': tf.io.FixedLenFeature([1], dtype=tf.int64, default_value=1)
      }
      example = tf.io.parse_single_example(example, feature_description)
      return example

  def _load_image_with_text(self, example):
      height = example['height']
      width = example['width']
      depth = example['depth']
      raw_image = example['masked_image']
      pxls = tf.io.parse_tensor(raw_image, out_type=tf.uint8)
      pxls = tf.reshape(pxls, shape=[height, width,depth])
      pxls = tf.image.resize(pxls, (cnt.INPUT_SIZE, cnt.INPUT_SIZE))
      pxls = tf.image.convert_image_dtype(pxls, tf.float32)

      length = example['text/length']
      text = example['text/string']

      label = self.indexing_fn(text, length)
      
      return pxls, label

  def indexing_fn(self, text, text_length):
      label = self.char_to_num(tf.strings.unicode_split(text, input_encoding="UTF-8"))
      label = tf.concat([[self.start_token], label, [self.end_token]], 0)
      label_length = tf.shape(label, tf.int64)[0]
      label = tf.pad(
        label, 
        paddings = [[0, self.max_length - label_length]], 
        constant_values = 0 # Pad with padding token
      )
      return label

  def tfrecord_ds(self, path_to_dataset, training: bool = True):
    if not isinstance(path_to_dataset, type(list)):
      path_to_dataset = [path_to_dataset]
    
    dataset = tf.data.TFRecordDataset(path_to_dataset)

    dataset = (dataset
      .map(self._parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)
      .map(self._load_image_with_text, num_parallel_calls=tf.data.AUTOTUNE)
      .cache()
    )
    
    if training:
      dataset = dataset.shuffle(2 * cnt.BATCH_SIZE)
    
    dataset = (dataset
      .batch(cnt.BATCH_SIZE, drop_remainder=True)
      .prefetch(tf.data.AUTOTUNE)
    )
    return dataset

  def tokens2texts(self, batch_tokens):
    batch_texts = []
    # Iterate over the results and get back the text
    for tokens in batch_tokens:
        indices = tf.gather(tokens, tf.where(tf.logical_and(
            tokens != 0, # For [PAD] token
            tokens != -1 # For blank label if use_ctc_decode 
        )))
        # Convert to string
        text = tf.strings.reduce_join(self.num_to_char(indices)) 
        text = text.numpy().decode('utf-8')
        text = text.replace(self.start_char, '').replace(self.end_char, '')
        batch_texts.append(text)
    return batch_texts 

### Synth dataset

  def _parse_synth_tfrecord_fn(self, example):    
    feature_description = {
      'height': tf.io.FixedLenFeature([], tf.int64),
      'width': tf.io.FixedLenFeature([], tf.int64),
      'depth': tf.io.FixedLenFeature([], tf.int64),
      'raw_image' : tf.io.FixedLenFeature([], tf.string),
      'text/string': tf.io.FixedLenFeature([], dtype=tf.string, default_value=''),
      'text/length': tf.io.FixedLenFeature([1], dtype=tf.int64, default_value=1)
    }
    example = tf.io.parse_single_example(example, feature_description)
    return example

  def _load_synth_image_with_text(self, example):
      height = example['height']
      width = example['width']
      depth = example['depth']
      raw_image = example['raw_image']

      pxls = tf.io.parse_tensor(raw_image, out_type=tf.uint8)
      pxls = tf.reshape(pxls, shape=[height, width, 3])
      pxls = tf.image.resize_with_pad(pxls, cnt.INPUT_SIZE, cnt.INPUT_SIZE)
      pxls = tf.image.convert_image_dtype(pxls, tf.float32)

      length = example['text/length']
      text = example['text/string']

      label = self.indexing_fn(text, length)
      
      return pxls, label

  def tfrecord_synth_ds(self, path_to_dataset, training: bool = True):
    if not isinstance(path_to_dataset, type(list)):
      path_to_dataset = [path_to_dataset]
    
    dataset = tf.data.TFRecordDataset(path_to_dataset)

    dataset = (dataset
      .map(self._parse_synth_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)
      .map(self._load_synth_image_with_text, num_parallel_calls=tf.data.AUTOTUNE)
      .cache()
    )
    
    if training:
      dataset = dataset.shuffle(2 * cnt.BATCH_SIZE)
    
    dataset = (dataset
      .batch(cnt.BATCH_SIZE, drop_remainder=True)
      .prefetch(tf.data.AUTOTUNE)
    )
    return dataset
