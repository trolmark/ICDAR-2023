import tensorflow as tf
from utils import update_tensor_column
import constants as cnt


class AONModel(tf.keras.Model):
  def __init__(self, loader, *args, **kwargs):
      super().__init__(*args, **kwargs)
      self.bccn = BCNNModule()
      self.aon_network = AONModule()
      self.loader = loader

      vocab_size = loader.char_to_num.vocabulary_size()
      units = 2**8
      embedding_dim = 256
      self.lstm = tf.keras.layers.Bidirectional(
          tf.keras.layers.RNN(tf.keras.layers.LSTMCell(units), return_sequences = True)
      )
      self.attention_decoder = RNNAttentionDecoder(embedding_dim, units, vocab_size)

  def call(self, images, training = False):
      features = self.bccn(images, training)
      assert features.get_shape()[1:] == (26, 26, 256)

      # AON
      features, clue = self.aon_network(features, training)
      assert features.get_shape()[1:] == (4, 23, 512)
      assert clue.get_shape()[1:] == (4, 23, 1)

      # FG
      features = tf.reduce_sum(features * clue, axis=1)
      features = tf.nn.tanh(features)
      assert features.get_shape()[1:] == (23, 512)

      # Decoder
      features = tf.transpose(features, [1, 0, 2], name='time_major')
      features = self.lstm(features)
      features = tf.transpose(features, [1, 0, 2], name='batch_major')
      return features

  @tf.function
  def _init_seq_tokens(self, batch_size, return_new_tokens=True):
      seq_tokens = tf.fill([batch_size, self.loader.max_length], self.loader.start_token)
      seq_tokens = tf.cast(seq_tokens, dtype=tf.int64)
      new_tokens = tf.fill([batch_size, 1], self.loader.start_token)
      new_tokens = tf.cast(new_tokens, dtype=tf.int64)

      seq_tokens = update_tensor_column(seq_tokens, new_tokens, 0)
      done = tf.zeros([batch_size, 1], dtype=tf.bool)
      if not return_new_tokens: return seq_tokens, done
      return seq_tokens, new_tokens, done

  @tf.function
  def _update_seq_tokens(self, y_pred, seq_tokens, done, pos_idx, return_new_tokens=True):
      # Set the logits for all masked tokens to -inf, so they are never chosen
      y_pred = tf.where(self.loader.token_mask, float('-inf'), y_pred)
      new_tokens = tf.argmax(y_pred, axis=-1) 

      # Add batch dimension if it is not present after argmax
      if tf.rank(new_tokens) == 1: new_tokens = tf.expand_dims(new_tokens, axis=1)

      # Once a sequence is done it only produces padding token
      new_tokens = tf.where(done, tf.constant(0, dtype=tf.int64), new_tokens)
      seq_tokens = update_tensor_column(seq_tokens, new_tokens, pos_idx)

      # If a sequence produces an `END_TOKEN`, set it `done` after that
      done = done | (new_tokens == self.loader.end_token)
      if not return_new_tokens: return seq_tokens, done
      return seq_tokens, new_tokens, done

  @tf.function
  def _update_metrics(self, batch):
      batch_images, batch_tokens = batch
      predictions = self.predict(batch_images) 
      self.compiled_metrics.update_state(batch_tokens, predictions)
      return {m.name: m.result() for m in self.metrics}

  @tf.function
  def _compute_loss_and_metrics(self, batch, is_training=False):
      x, y = batch
      batch_size = cnt.BATCH_SIZE

      loss = 0
      hidden = self.attention_decoder.reset_state(batch_size=batch_size)
      dec_input = tf.expand_dims([self.loader.start_token] * batch_size, 1) 
      features = self(x, training=is_training)

      for i in range(1, self.loader.max_length):
        predictions, hidden, _ = self.attention_decoder(
          dec_input, features, hidden, training=is_training
        )
        loss += self.compiled_loss(y[:, i], predictions)
        dec_input = tf.expand_dims(y[:, i], 1)
      
      total_loss = (loss / int(self.loader.max_length))
      metrics = self._update_metrics(batch)
      return total_loss, {'loss': total_loss, **metrics}

  @tf.function
  def train_step(self, data):
      with tf.GradientTape() as tape:
          loss, display_results = self._compute_loss_and_metrics(data, is_training=True)

      trainable_vars = self.trainable_variables
      gradients = tape.gradient(loss, trainable_vars)
      self.optimizer.apply_gradients(zip(gradients, trainable_vars))
      return display_results
    
  @tf.function
  def test_step(self, batch):
      _, display_results = self._compute_loss_and_metrics(batch, is_training=False)
      return display_results

  @tf.function
  def predict(self, batch_images):
      batch_size = cnt.BATCH_SIZE
      seq_tokens, new_tokens, done = self._init_seq_tokens(batch_size)
    
      hidden = self.attention_decoder.reset_state(batch_size=batch_size)
      dec_input = tf.expand_dims([1] * batch_size, 1)
      enc_output = self(batch_images, training=False)

      for i in range(1, self.loader.max_length):
          y_pred, hidden, attention_weights = self.attention_decoder(new_tokens, enc_output, hidden, training=False)
          seq_tokens, new_tokens, done = self._update_seq_tokens(y_pred, seq_tokens, done, i)
          if tf.executing_eagerly() and tf.reduce_all(done): break

      return seq_tokens


class BahdanauAttention(tf.keras.Model):
  """Bahdanau Attention Layer.

  Attributes:
    w1: weights that process the feature
    w2: weights that process the memory state
    v: projection layer that project score vector to scalar
  """

  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.w1 = tf.keras.layers.Dense(units)
    self.w2 = tf.keras.layers.Dense(units)
    self.v = tf.keras.layers.Dense(1)

  def call(self, features, hidden):
    # features(CNN_encoder output) shape == (batch_size, 64, embedding_dim)

    # hidden shape == (batch_size, hidden_size)
    # hidden_with_time_axis shape == (batch_size, 1, hidden_size)
    hidden_with_time_axis = tf.expand_dims(hidden, 1)

    # score shape == (batch_size, 64, hidden_size)
    score = tf.nn.tanh(self.w1(features) + self.w2(hidden_with_time_axis))

    # attention_weights shape == (batch_size, 64, 1)
    # you get 1 at the last axis because you are applying score to self.V
    attention_weights = tf.nn.softmax(self.v(score), axis=1)

    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * features
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights


class RNNAttentionDecoder(tf.keras.Model):
  """Decoder that decodes a embedded representation.

  Attributes:
    units: size of the hidden units of GRU
    embedding: embedding matrix for text
    gru: the GRU layer
    fc1: first dense layer that process tha hidden state
    fc2: second dense layer that process tha hidden state
    dropout: dropout layer applied to the hidden state
    attention: Bahdanau attention layer
  """

  def __init__(self, embedding_dim, units, vocab_size, name='RNNDecoder'):
    super(RNNAttentionDecoder, self).__init__(name=name)
    self.units = units

    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(
        self.units,
        return_sequences=True,
        return_state=True,
        recurrent_initializer='glorot_uniform')
    self.fc1 = tf.keras.layers.Dense(self.units)
    self.fc2 = tf.keras.layers.Dense(vocab_size)
    self.dropout = tf.keras.layers.Dropout(0.5)
    self.attention = BahdanauAttention(self.units)

  def call(self, x, features, hidden, training=None):
    # defining attention as a separate model
    context_vector, attention_weights = self.attention(features, hidden)

    # x shape after passing through embedding == (batch_size, 1, embedding_dim)
    x = self.embedding(x)

    # x shape after concatenation == (batch_size,1, embedding_dim + hidden_size)
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

    # passing the concatenated vector to the GRU
    output, state = self.gru(x)

    # shape == (batch_size, max_length, hidden_size)
    x = self.fc1(output)

    # x shape == (batch_size * max_length, hidden_size)
    x = tf.reshape(x, (-1, x.shape[2]))

    # output shape == (batch_size * max_length, vocab)
    x = self.fc2(x)

    state = self.dropout(state, training=training)
    return x, state, attention_weights

  def reset_state(self, batch_size):
    return tf.zeros((batch_size, self.units))


class BCNNModule(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

        conv_layer = tf.keras.layers.Conv2D
        self.conv1 = tf.keras.Sequential([
            conv_layer(64, 3, (1, 1), 'same', use_bias = False, name = "conv1"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU()
        ])
        self.pool1 = tf.keras.layers.MaxPool2D((2, 2), strides=2, padding="valid", name = "pool1")

        self.conv2 = tf.keras.Sequential([
            conv_layer(128, 3, (1, 1), 'same', use_bias = False, name = "conv2"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU()
        ])
        self.pool2 = tf.keras.layers.MaxPool2D((2, 2), strides=2, padding="valid", name = "pool2")

        self.conv3 = tf.keras.Sequential([
            conv_layer(256, 3, (1, 1), 'same', use_bias = False, name = "conv3"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU()
        ])
        self.conv4 = tf.keras.Sequential([
            conv_layer(256, 3, (1, 1), 'same', use_bias = False, name = "conv4"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU()
        ])

    def call(self, inputs, training = False):
        x = self.conv1(inputs, training)
        x = self.pool1(x)
        x = self.conv2(x, training)
        x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]])
        x = self.pool2(x)
        x = self.conv3(x, training)
        x = self.conv4(x, training)
        return x


class SharedCNN(tf.keras.layers.Layer):
    def __init__(self):
      super().__init__()
      conv_layer = tf.keras.layers.Conv2D
      self.conv1 = tf.keras.Sequential([
          conv_layer(512, 3, (1, 1), 'same', use_bias = False, name = "conv1"),
          tf.keras.layers.BatchNormalization(),
          tf.keras.layers.ReLU()
      ])
      self.pool1 = tf.keras.layers.MaxPool2D((2, 2), strides=(2, 1), padding="valid", name = "pool1")

      self.conv2 = tf.keras.Sequential([
          conv_layer(512, 3, (1, 1), 'same', use_bias = False, name = "conv2"),
          tf.keras.layers.BatchNormalization(),
          tf.keras.layers.ReLU()
      ])
      self.pool2 = tf.keras.layers.MaxPool2D((2, 2), strides=(2, 1), padding="valid", name = "pool2")

      self.conv3 = tf.keras.Sequential([
          conv_layer(512, 3, (1, 1), 'same', use_bias = False, name = "conv3"),
          tf.keras.layers.BatchNormalization(),
          tf.keras.layers.ReLU()
      ])
      self.pool3 = tf.keras.layers.MaxPool2D((2, 2), strides=(2, 1), padding="valid", name = "pool3")

      self.conv4 = tf.keras.Sequential([
          conv_layer(512, 3, (1, 1), 'same', use_bias = False, name = "conv4"),
          tf.keras.layers.BatchNormalization(),
          tf.keras.layers.ReLU()
      ])
      self.pool4 = tf.keras.layers.MaxPool2D((2, 2), strides=(2, 1), padding="valid", name = "pool4")

      self.conv5 = tf.keras.Sequential([
          conv_layer(512, 3, (1, 1), 'same', use_bias = False, name = "conv5"),
          tf.keras.layers.BatchNormalization(),
          tf.keras.layers.ReLU()
      ])
      self.pool5 = tf.keras.layers.MaxPool2D((2, 2), strides=(2, 1), padding="valid", name = "pool5")


    def call(self, inputs, training = False):
      x = self.conv1(inputs, training)
      x = tf.pad(x, [[0, 0], [1, 1], [0, 0], [0, 0]])
      x = self.pool1(x)

      x = self.conv2(x, training)
      x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]])
      x = self.pool2(x)

      x = self.conv3(x, training)
      x = self.pool3(x)

      x = self.conv4(x, training)
      x = self.pool4(x)
      x = self.conv5(x, training)
      x = self.pool5(x)
      x = tf.reshape(x, (-1, 23, 512))
      return x



class ClueModule(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        conv_layer = tf.keras.layers.Conv2D
        self.conv1 = tf.keras.Sequential([
          conv_layer(512, 3, (1, 1), 'same', use_bias = False, name = "conv1"),
          tf.keras.layers.BatchNormalization(),
          tf.keras.layers.ReLU()
        ])
        self.pool1 = tf.keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding="valid", name = "pool1")

        self.conv2 = tf.keras.Sequential([
            conv_layer(512, 3, (1, 1), 'same', use_bias = False, name = "conv2"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU()
        ])
        self.pool2 = tf.keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding="valid", name = "pool2")

        self.linear1 = tf.keras.layers.Dense(23, activation = "relu")
        self.dropout = tf.keras.layers.Dropout(0.8)
        self.linear2 = tf.keras.layers.Dense(4, activation = "softmax")

    def call(self, inputs, training = False):
        assert inputs.get_shape()[1:] == (26, 26, 256)
        x = self.conv1(inputs, training)
        x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]])
        x = self.pool1(x)

        x = self.conv2(x, training)
        x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]])
        x = self.pool2(x)

        x = tf.reshape(x, (-1, 64, 512))
        x = tf.transpose(x, perm=[0, 2, 1])

        x = self.linear1(x)
        x = self.dropout(x, training)
        x = tf.transpose(x, perm=[0, 2, 1])

        x = self.linear2(x)
        x = tf.transpose(x, perm=[0, 2, 1])
        x = tf.expand_dims(x, axis=-1)

        return x

class AONModule(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.shared_cnn = SharedCNN()

        rnn_size = 2**8
        self.hfeaturesLSTM = tf.keras.layers.Bidirectional(
          tf.keras.layers.RNN(tf.keras.layers.LSTMCell(rnn_size), return_sequences = True)
        )
        self.vfeaturesLSTM = tf.keras.layers.Bidirectional(
          tf.keras.layers.RNN(tf.keras.layers.LSTMCell(rnn_size), return_sequences = True)
        )
        self.clue_network = ClueModule()
    
    def call(self, inputs, training = False):

        assert inputs.get_shape()[1:] == (26, 26, 256)

        hfeatures = self.shared_cnn(inputs, training)
        hfeatures = tf.transpose(hfeatures, perm=[1, 0, 2], name='h_time_major')
        hfeatures = self.hfeaturesLSTM(hfeatures)
        
        vfeatures = self.shared_cnn(tf.image.rot90(inputs), training)
        vfeatures = tf.transpose(vfeatures, perm=[1, 0, 2], name='v_time_major')
        vfeatures = self.vfeaturesLSTM(vfeatures)

        features = (hfeatures, tf.reverse(hfeatures, axis=[0])) + \
                   (vfeatures, tf.reverse(vfeatures, axis=[0]))
        features = tf.stack(features, axis=1)
        features = tf.transpose(features, [2, 1, 0, 3])

        clue = self.clue_network(inputs, training)

        return features, clue