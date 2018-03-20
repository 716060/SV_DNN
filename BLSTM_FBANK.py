'''
A Bidirectional Recurrent Neural Network (LSTM) implementation example using TensorFlow library.
Using Bidirectional Rcurrent Neural Network (LSTM) to train an embedding system for speaker verification
'''

from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn
import sidekit
import list_file_parse as LFP
import numpy as np

# prepare labels and data    training_list_all_less training_validation_list_all_less1 training_list_all_mixer
list_file = '../Trianing_triplet_small_data/evaluation_list/training_list_all_mixer_exist_FBANK_small.scp'

traning_file_name_dir, traning_spk_name = LFP.list_file_parse(list_file)

spk_name_uniq, spk_name_index, spk_name_index_inv, spk_name_counts = np.unique(traning_spk_name, return_index=True, return_inverse=True, return_counts=True)
spk_name_index_inv_traning = spk_name_index_inv

index_train = np.linspace(0, traning_spk_name.__len__()-1, num = traning_spk_name.__len__())

ind = tf.placeholder(tf.uint8, [spk_name_index_inv.__len__()])
lbl_one_hot = tf.one_hot(ind, spk_name_uniq.__len__(), 1.0, 0.0, dtype='float')


'''
To classify images using a bidirectional recurrent neural network, we consider
every image row as a sequence of pixels. Because MNIST image shape is 28*28px,
we will then handle 28 sequences of 28 steps for every sample.
'''

# Parameters
learning_rate = 0.001
training_iters = 5000
batch_size = 128
display_step = 1
epoch = 2000

# Network Parameters
n_input = 40 # MNIST data input (img shape: 28*28)

n_hidden = 512 # hidden layer num of features
n_classes = spk_name_uniq.__len__() # MNIST total classes (0-9 digits)
duration = 2
frame_per_second = 100
n_steps = duration*frame_per_second # timesteps

# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])

# Define weights
weights = {
    # Hidden layer weights => 2*n_hidden because of forward + backward cells
    'out': tf.Variable(tf.random_normal([2*n_hidden, n_hidden]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_hidden]))
}

# Define weights
weights_soft = {
    # Hidden layer weights => 2*n_hidden because of forward + backward cells
    'classify': tf.Variable(tf.random_normal([2*n_hidden, n_classes]))
}
biases_soft = {
    'classify': tf.Variable(tf.random_normal([n_classes]))
}


def BiRNN(x, weights, biases):

    # Prepare data shape to match `bidirectional_rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, n_steps, 1)

    # Define lstm cells with tensorflow
    # Forward direction cell
    lstm_fw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)    #LayerNormBasicLSTMCell
    # Backward direction cell
    lstm_bw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)

    # Get lstm cell output
    try:
        outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                              dtype=tf.float32)
    except Exception: # Old TensorFlow version only returns outputs not states
        outputs = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                        dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    # return tf.matmul(outputs[-1], weights['out']) + biases['out']
    return outputs


def load_features(file_names):

    features = []
    for file_name in file_names:
        # segment = RamdonSegment(file_name_dir, self.duration)
        d1, fp, dt, tc, t = sidekit.frontend.io.read_htk(file_name, frame_per_second=frame_per_second)
        features.append(d1)
    return features




def generate_training_samples_from_loaded_features(features):

    for d1 in features:
        if d1.shape[0] > duration * frame_per_second:

            random_start = np.random.randint(0, d1.shape[0] - duration * frame_per_second)
            # random_start = 0
            segment = d1[random_start:random_start + duration * frame_per_second, :]
        else:

            a = np.zeros((duration * frame_per_second - d1.shape[0], 40))
            segment = np.concatenate((d1, a), axis=0)  # padding zeros

        yield (segment)



embedding = BiRNN(x, weights, biases)[-1]
# pred = tf.matmul(lstm_out, weights['out']) + biases['out']
# pred = BiRNN(x, weights, biases)
# embedding = tf.nn.relu(tf.matmul(lstm_out, weights['out']) + biases['out'])
# embedding = tf.nn.l2_normalize(embedding, dim = 0)
pred = tf.matmul(embedding, weights_soft['classify']) + biases_soft['classify']

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))



# initial saver
saver = tf.train.Saver()
# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    labels = sess.run([lbl_one_hot], feed_dict={ind: spk_name_index_inv})
    labels = np.asarray(labels)[0]

    # segmentclass1 = generate_triplet_from_list_file(traning_file_name_dir)
    features = load_features(traning_file_name_dir)
    segmentclass1 = generate_training_samples_from_loaded_features(features)
# split data
    Segment = []
    counter = 0
    for segment in segmentclass1:
        Segment.append(segment)
        # print(counter)
        counter += 1
    Segment = np.asarray(Segment)
    np.random.shuffle(index_train)
    index_all = index_train.astype(int)
    index_valid = index_all[0:500]
    index_train_select = index_all[500:]

    Segment_valid = Segment[index_valid]
    labels_valid = labels[index_valid]

    labels_train_select = labels[index_train_select]

    features_traning = []
    for num in index_train_select:
        features_traning.append(features[num])
    features = None
    # Keep training until reach max iterations
    # obtain file dir and spk name
    index_train = np.linspace(0, index_train_select.__len__() - 1, num=index_train_select.__len__())
    for num in range(0,epoch-1):
        step = 1
        segmentclass1 = generate_training_samples_from_loaded_features(features_traning)

        Segment_train = []
        counter = 0
        for segment in segmentclass1:
            Segment_train.append(segment)
            # print(counter)
            counter += 1
        Segment_train = np.asarray(Segment_train)

        # shuffle training data
        np.random.shuffle(index_train)
        index_train = index_train.astype(int)


        labels_train = labels_train_select[index_train]
        Segment_train = Segment_train[index_train]

        while step * batch_size < Segment_train.shape[0]:
            # Run optimization op (backprop)
            batch_x = Segment_train[(step - 1) * batch_size:step * batch_size]
            batch_y = labels_train[(step - 1) * batch_size:step * batch_size]
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})

            acc = sess.run(accuracy, feed_dict={x: Segment_valid, y: labels_valid})
            print("Testing Accuracy: "+"{:.5f}".format(acc))

            if step % display_step == 0:
                # Calculate batch accuracy
                acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
                # Calculate batch loss
                loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
                print("Iter " + str(step * batch_size+num*step * batch_size) + ", Minibatch Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))
            step += 1

        print("Testing Accuracy:", \
              sess.run(accuracy, feed_dict={x: Segment_valid, y: labels_valid}))

    print("Optimization Finished!")

    save_path = saver.save(sess, "/media/jianboma/Jianbo_Backup1/LSTM_Short_Duration/check/model_2s_FBANK.ckpt")
    print("Model saved in file: %s" % save_path)


