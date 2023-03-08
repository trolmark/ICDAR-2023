import os
import tensorflow as tf

def update_tensor_column(tensor, values, col_idx):
    if col_idx < 0: raise ValueError("col_idx must be >= 0")
    rows = tf.range(tf.shape(tensor)[0])
    column = tf.zeros_like(rows) + col_idx
    idxs = tf.stack([rows, column], axis=1)
    return tf.tensor_scatter_nd_update(tensor, idxs, tf.squeeze(values, axis=-1))

def load_charset(charset_path):
    """ Load character set.
    """
    if os.path.exists(charset_path):
        _, ext = os.path.splitext(charset_path)
        with open(charset_path) as f:
          charset = f.read().splitlines()
    else:
        raise NotImplementedError

    return charset