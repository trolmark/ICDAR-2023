import tensorflow as tf
import tensorflow.keras.backend as K

import constants


class WeightedMSELoss(tf.keras.losses.Loss):

  def call(self, y_true, y_pred):
      y_pred = tf.convert_to_tensor(y_pred)
      y_true = tf.cast(y_true, y_pred.dtype)
      """
      apply weights on heatmap mse loss to only pick valid keypoint heatmap
      since y_true would be gt_heatmap with shape
      (batch_size, heatmap_size[0], heatmap_size[1], num_keypoints)
      we sum up the heatmap for each keypoints and check. Sum for invalid
      keypoint would be 0, so we can get a keypoint weights tensor with shape
      (batch_size, 1, 1, num_keypoints)
      and multiply to loss
      """
      heatmap_sum = K.sum(K.sum(y_true, axis=1, keepdims=True), axis=2, keepdims=True)

      # keypoint_weights shape: (batch_size, 1, 1, num_keypoints), with
      # valid_keypoint = 1.0, invalid_keypoint = 0.0
      keypoint_weights = 1.0 - K.cast(K.equal(heatmap_sum, 0.0), 'float32')
      return K.sqrt(K.mean(K.square((y_true - y_pred) * keypoint_weights)))
