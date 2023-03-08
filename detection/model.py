import tensorflow as tf
from tensorflow.keras import backend
import constants

class FeaturePyramid(tf.keras.layers.Layer):
    """Builds the Feature Pyramid with the feature maps from the backbone."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        depth = 128
        self.conv_c3_1x1 = tf.keras.layers.Conv2D(depth, 1, 1, "same")
        self.conv_c4_1x1 = tf.keras.layers.Conv2D(depth, 1, 1, "same")
        self.conv_c5_1x1 = tf.keras.layers.Conv2D(depth, 1, 1, "same")
        self.conv_c3_3x3 = tf.keras.layers.Conv2D(depth, 3, 1, "same")
        self.conv_c4_3x3 = tf.keras.layers.Conv2D(depth, 3, 1, "same")
        self.conv_c5_3x3 = tf.keras.layers.Conv2D(depth, 3, 1, "same")
        self.conv_c6_3x3 = tf.keras.layers.Conv2D(depth, 3, 2, "same")
        self.upsample_2x = tf.keras.layers.UpSampling2D(2)

    def call(self, inputs, training=False):
        c3_output, c4_output, c5_output = inputs
        p3_output = self.conv_c3_1x1(c3_output, training=training)
        p4_output = self.conv_c4_1x1(c4_output, training=training)
        p5_output = self.conv_c5_1x1(c5_output, training=training)
        p4_output = p4_output + self.upsample_2x(p5_output, training=training)
        p3_output = p3_output + self.upsample_2x(p4_output, training=training)
        p3_output = self.conv_c3_3x3(p3_output, training=training)
        p4_output = self.conv_c4_3x3(p4_output, training=training)
        p5_output = self.conv_c5_3x3(p5_output, training=training)
        p6_output = self.conv_c6_3x3(c5_output, training=training)

        return self.upsample_2x(p3_output +\
          self.upsample_2x(p4_output) +\
          self.upsample_2x(self.upsample_2x(p5_output)) +\
          self.upsample_2x(self.upsample_2x(self.upsample_2x(p6_output))))

class PredictionHead(tf.keras.models.Model):
  def __init__(self, num_classes, out_activation, depth = 128, **kwargs):
        super(PredictionHead, self).__init__(name="PredictionHead", **kwargs)
        self.num_classes = num_classes

        conv_2d_op = tf.keras.layers.Conv2D

        self.conv_3x3 = conv_2d_op(depth, kernel_size=3, strides=1, padding="same")
        self.conv_1x1 = conv_2d_op(depth, kernel_size=1, strides=1, padding="same")
        self.conv_out = tf.keras.layers.Conv2D(num_classes, kernel_size=1, strides=1, padding="same", activation=out_activation)

  def call(self, images, training = False):
    x = self.conv_3x3(images, training=training)
    x = self.conv_1x1(x, training=training)
    x = tf.nn.relu6(x)
    x = self.conv_out(x, training=training)
    return x


def _build_backbone_mobilenetV2(backbone_weights, num_input_channels):
  input_shape = (constants.INPUT_SIZE, constants.INPUT_SIZE, num_input_channels)
  inputs = tf.keras.layers.Input(shape=input_shape)
  x = inputs

  full_mobilenet_v2 = tf.keras.applications.MobileNetV2(
      input_shape=input_shape, 
      include_top=False,
      alpha=0.75,
      input_tensor=x, weights=backbone_weights
  )

  layer_names = ['block_5_project_BN', 'block_9_project_BN', 'block_15_project_BN']
  layer_outputs = [full_mobilenet_v2.get_layer(name).output for name in layer_names]
  backbone = tf.keras.models.Model(inputs=full_mobilenet_v2.input, outputs=layer_outputs, name=full_mobilenet_v2.name)
  return backbone


class CornerNet(tf.keras.models.Model):
  """
  Single-head model, which predict corners with heatmaps
  """
  def __init__(self, num_classes, backbone,  **kwargs):
        super(CornerNet, self).__init__(name="CornerNet", **kwargs)
        self.backbone = backbone
        self.fpn = FeaturePyramid()
        self.heatmap_head = PredictionHead(num_classes = num_classes, out_activation = "sigmoid")
        self.loss_metric = tf.keras.metrics.Mean(name="loss")

  @property
  def metrics(self):
      return super().metrics + self.train_metrics

  @property
  def train_metrics(self):
      return [
          self.loss_metric
      ]

  def call(self, images, training = False):
      backbone_features = self.backbone(images, training)
      fpn_features = self.fpn(backbone_features, training)
      
      heatmap_keypoints = self.heatmap_head(fpn_features, training)
      return heatmap_keypoints

  def compile(self, heatmap_loss=None, loss=None, metrics=None, **kwargs):
      super().compile(metrics=metrics, **kwargs)
      if loss is not None:
          raise ValueError(
              "`KeypointNet` does not accept a `loss` to `compile()`. "
              "Instead, please pass `heatmap_loss` "
              "`loss` will be ignored during training."
          )
      self.corner_heatmap_loss = heatmap_loss
  
  def _backward(self, y_true, y_pred):
      loss = self.corner_heatmap_loss(
          y_true,
          y_pred,
      )
      self.loss_metric.update_state(loss)
      return loss
     
  def train_step(self, data):
      x, y_true = data
     
      with tf.GradientTape() as tape:
          y_pred = self(x, training=True) 
          loss = self._backward(y_true, y_pred)

      trainable_vars = self.trainable_variables
      gradients = tape.gradient(loss, trainable_vars)
      self.optimizer.apply_gradients(zip(gradients, trainable_vars))

      self.compiled_metrics.update_state(y_true, y_pred)
      return {m.name: m.result() for m in self.metrics}

  def test_step(self, data):
      x, y_true = data
      
      y_pred = self(x, training=False)
      loss = self._backward(y_true, y_pred)

      self.compiled_metrics.update_state(y_true, y_pred)
      return {m.name: m.result() for m in self.metrics}

  