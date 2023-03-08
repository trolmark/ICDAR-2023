import tensorflow as tf
from tensorflow.keras.losses import SparseCategoricalCrossentropy

class MaskedLoss(tf.keras.losses.Loss):
    def __init__(self, name='masked_loss', **kwargs):
        super(MaskedLoss, self).__init__(name=name, **kwargs)
        # The padding token need to be 0 for SparseCategoricalCrossentropy
        # See https://stackoverflow.com/questions/63171001 if loss == nan
        self.loss = SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    def call(self, y_true, y_pred):
        loss = self.loss(y_true, y_pred)
        mask = tf.cast(y_true != 0, tf.float32)
        return tf.math.divide_no_nan(tf.reduce_sum(loss * mask), tf.reduce_sum(mask))