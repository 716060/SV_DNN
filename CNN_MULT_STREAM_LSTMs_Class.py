
import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn

class CNN_MULT_STREAM_LSTMs:
    def __init__(self, inputs, num_class, weights=None, sess=None):
        self.inputs = inputs
        self.num_class = num_class
        self.convlayers()

        self.blstm_layers()
        self.fc_layers()


        if weights is not None and sess is not None:
            self.load_weights(weights, sess)


    def convlayers(self):
        self.parameters = []

        with tf.name_scope('preprocess') as scope:
            self.inputs = tf.reshape(self.inputs, shape=[-1, 312, 39, 1])

        # conv1_1
        with tf.name_scope('conv1_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([5, 5, 1, 6], dtype=tf.float32,
                                                     stddev=0.01), name='weights')
            conv = tf.nn.conv2d(self.inputs, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.truncated_normal(shape=[6], dtype=tf.float32, stddev=0.01),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv1_2
        with tf.name_scope('conv1_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([5, 5, 6, 6], dtype=tf.float32,
                                                     stddev=0.01), name='weights')
            conv = tf.nn.conv2d(self.conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.truncated_normal(shape=[6], dtype=tf.float32, stddev=0.01),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool1
        self.pool1 = tf.nn.max_pool(self.conv1_1, #conv1_2
                               ksize=[1, 2, 2, 1],#[1, 2, 2, 1]
                               strides=[1, 2, 2, 1],#[1, 2, 2, 1]
                               padding='SAME',
                               name='pool1')

        # conv2_1
        with tf.name_scope('conv2_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 6, 16], dtype=tf.float32,
                                                     stddev=0.01), name='weights')
            conv = tf.nn.conv2d(self.pool1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.truncated_normal(shape=[16], dtype=tf.float32, stddev=0.01),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv2_2
        with tf.name_scope('conv2_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 16, 16], dtype=tf.float32,
                                                     stddev=0.01), name='weights')
            conv = tf.nn.conv2d(self.conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.truncated_normal(shape=[16], dtype=tf.float32, stddev=0.01),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool2
        self.pool2 = tf.nn.max_pool(self.conv2_1,#conv2_2
                               ksize=[1, 2, 2, 1],#[1, 2, 2, 1]
                               strides=[1, 2, 2, 1],#[1, 2, 2, 1]
                               padding='SAME',
                               name='pool2')

    def blstm_layers(self):

        self.out_lstm = []
        # Forward direction cell
        lstm_fw_cell = rnn.BasicLSTMCell(128, forget_bias=1.0)  # LayerNormBasicLSTMCell
        # Backward direction cell
        lstm_bw_cell = rnn.BasicLSTMCell(128, forget_bias=1.0)

        for i in range(0, self.pool2._shape._dims[3]._value - 1):

            if i is 0:
                x = tf.unstack(self.pool2[:, :, :, i], self.pool2[:, :, :, i].shape[1], 1)
                with tf.variable_scope('lstm_' + str(i)) as scope:
                    try:
                        lstm_outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                                                          dtype=tf.float32)
                    except Exception:  # Old TensorFlow version only returns outputs not states
                        lstm_outputs = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                                                   dtype=tf.float32)
                    self.out_lstm = lstm_outputs[-1]
                    self.parameters +=  tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name)

            else:
                x = tf.unstack(self.pool2[:, :, :, i], self.pool2[:, :, :, i].shape[1], 1)
                with tf.variable_scope('lstm_' + str(i)) as scope:
                    # Forward direction cell
                    lstm_fw_cell = rnn.BasicLSTMCell(128, forget_bias=1.0)  # LayerNormBasicLSTMCell
                    # Backward direction cell
                    lstm_bw_cell = rnn.BasicLSTMCell(128, forget_bias=1.0)
                    try:
                        lstm_outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                                                          dtype=tf.float32)
                    except Exception:  # Old TensorFlow version only returns outputs not states
                        lstm_outputs = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                                                    dtype=tf.float32)
                    self.out_lstm = tf.concat([self.out_lstm, lstm_outputs[-1]], axis=1)
                    self.parameters += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name)





    def fc_layers(self):
        # fc1
        with tf.name_scope('fc1') as scope:
            shape = int(np.prod(self.out_lstm.get_shape()[1:]))
            fc1w = tf.Variable(tf.truncated_normal([shape, 1024],
                                                         dtype=tf.float32,
                                                         stddev=0.01), name='weights')
            fc1b = tf.Variable(tf.truncated_normal(shape=[1024], dtype=tf.float32, stddev=0.01),
                                 trainable=True, name='biases')
            # pool5_flat = tf.reshape(self.pool5, [-1, shape])
            fc1l = tf.nn.bias_add(tf.matmul(self.out_lstm, fc1w), fc1b)
            #add batch_normalization
            fc1l = tf.layers.batch_normalization(fc1l,
                                                 center = True, scale = True)
            self.fc1 = tf.nn.relu(fc1l)
            self.parameters += [fc1w, fc1b]

        # fc2
        # with tf.name_scope('fc2') as scope:
        #     fc2w = tf.Variable(tf.truncated_normal([4096, 400],
        #                                                  dtype=tf.float32,
        #                                                  stddev=0.01), name='weights')
        #     fc2b = tf.Variable(tf.truncated_normal(shape=[400], dtype=tf.float32, stddev=0.01),
        #                          trainable=True, name='biases')
        #     fc2l = tf.nn.bias_add(tf.matmul(self.fc1, fc2w), fc2b)
        #     # add batch_normalization
        #     fc2l = tf.layers.batch_normalization(fc2l,
        #                                          center=True, scale=True)
        #     self.fc2 = tf.nn.relu(fc2l)
        #     self.parameters += [fc2w, fc2b]

        # fc3
        with tf.name_scope('fc3') as scope:
            fc3w = tf.Variable(tf.truncated_normal([1024, self.num_class],
                                                         dtype=tf.float32,
                                                         stddev=0.01), name='weights')

            self.fc3l = tf.matmul(self.fc1, fc3w)

            self.parameters += [fc3w]

    def load_weights(self, weight_file, sess):
        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        for i, k in enumerate(keys):
            print(i, k, np.shape(weights[k]))
            sess.run(self.parameters[i].assign(weights[k]))

    def loss(self, logits, labels):
        """Add L2Loss to all the trainable variables.
          Add summary for "Loss" and "Loss/avg".
          Args:
            logits: Logits from inference().
            labels: Labels from distorted_inputs or inputs(). 1-D tensor
                    of shape [batch_size]
          Returns:
            Loss tensor of type float.
          """
        # Calculate the average cross entropy loss across the batch.
        labels = tf.cast(labels, tf.int64)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits, name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        tf.add_to_collection('losses', cross_entropy_mean)

        # The total loss is defined as the cross entropy loss plus all of the weight
        # decay terms (L2 loss).
        return tf.add_n(tf.get_collection('losses'), name='total_loss')

    def Length(self, sequence):
        used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
        length = tf.reduce_sum(used, 1)
        length = tf.cast(length, tf.int32)
        return length

    # Batch the variable length tensor with dynamic padding
    # should I generate samples from first layer and store them first
    def last_relevant(self, output, length, n_hidden_lstm):

        batch_size = tf.shape(output)[0]
        # batch_size = 128
        max_length = tf.shape(output)[1]

        # out_size = int(output.get_shape()[2])
        # max_length = n_steps
        # output = tf.concat([output[0],output[1]],2)
        out_size = n_hidden_lstm * 2
        index = tf.range(0, batch_size) * max_length + (length - 1)
        flat = tf.reshape(output, [-1, out_size])
        relevant = tf.gather(flat, index)
        return relevant