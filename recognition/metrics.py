import tensorflow as tf

class CharacterAccuracy(tf.keras.metrics.Metric):
    def __init__(
        self,
        name = 'char_acc', 
        **kwargs
    ):
        super(CharacterAccuracy, self).__init__(name=name, **kwargs)
        self.count = self.add_weight(name='count', initializer='zeros')
        self.total = self.add_weight(name='total', initializer='zeros')
                
    def update_state(self, y_true, y_pred, **kwargs):
        batch_size, max_length = tf.shape(y_true)[0], tf.shape(y_true)[1]
        
        num_errors = tf.logical_and(y_true != y_pred, y_true != 0)
        num_errors = tf.reduce_sum(tf.cast(num_errors, tf.float32))
        total = tf.reduce_sum(tf.cast(y_true != 0, tf.float32))

        self.count.assign_add(total - num_errors)
        self.total.assign_add(total)

    def result(self):
        return tf.math.divide_no_nan(self.count, self.total)

    def reset_state(self):
        # The state of the metric will be reset at the start of each epoch.
        self.count.assign(0)
        self.total.assign(0)