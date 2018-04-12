
'''
a CNN with a stream of Bi-LSTMs are used to produce a embedding for speaker verification
Author: Jianbo Ma
Project:
'''

from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn
import sidekit
import list_file_parse as LFP
import numpy as np
import re
import os.path
import time
from datetime import datetime
from six.moves import xrange
import scipy.io as sio
import CNN_MULT_STREAM_LSTMs_Class as CNN_LSTMs
import feature_prepare
import CNN_LSTM_Parameters as CNN_LSTM_P
import batch_generator_CNN_LSTMs as BG_CNN_LSTMs


#male_TmatrixListfile_100spk
# prepare labels and data    second_layer_training_list_all_mixer training_list_all_less training_validation_list_all_less1 training_list_all_mixer Male_8conv-10sec_testParsed_MFCC_MSR
list_file = '../Trianing_triplet_small_data/evaluation_list/male_TmatrixListfile_100spk.scp'
_, _, spk_name_uniq,_,_ = feature_prepare.create_labels(list_file)
n_classes = spk_name_uniq.__len__()
num_gpus = 1

# FLAGS = tf.app.flags.FLAGS
#
# tf.app.flags.DEFINE_string('train_dir', '/tmp/cifar10_train',
#                            """Directory where to write event logs """
#                            """and checkpoint.""")
# tf.app.flags.DEFINE_integer('max_steps', 1000000,
#                             """Number of batches to run.""")
# tf.app.flags.DEFINE_integer('num_gpus', 1,
#                             """How many GPUs to use.""")
# tf.app.flags.DEFINE_boolean('log_device_placement', False,
# """Whether to log device placement.""")


def tower_loss(scope, input_features, labels, n_classes):
    """Calculate the total loss on a single tower running the CIFAR model.
      Args:
    	scope: unique prefix string identifying the CIFAR tower, e.g. 'tower_0'
    	images: Images. 4D tensor of shape [batch_size, height, width, 3].
    	labels: Labels. 1D tensor of shape [batch_size].
      Returns:
    	 Tensor of shape [] containing the total loss for a batch of data
      """

    # Build inference Graph.

    out_CNN_LSTMs = CNN_LSTMs.CNN_MULT_STREAM_LSTMs(input_features, n_classes, None, None)
    # Build the portion of the Graph calculating the losses. Note that we will
    # assemble the total_loss using a custom function below.
    _ = CNN_LSTMs.loss(out_CNN_LSTMs, labels)

    # Assemble all of the losses for the current tower only.
    losses = tf.get_collection('losses', scope)

    # Calculate the total loss for the current tower.
    total_loss = tf.add_n(losses, name='total_loss')

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
        # session. This helps the clarity of presentation on tensorboard.
        loss_name = re.sub('%s_[0-9]*/' % CNN_LSTM_P.TOWER_NAME, '', l.op.name)
        tf.summary.scalar(loss_name, l)

        return total_loss



def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
      Note that this function provides a synchronization point across all towers.
      Args:
        tower_grads: List of lists of (gradient, variable) tuples. The outer list
          is over individual gradients. The inner list is over the gradient
          calculation for each tower.
      Returns:
         List of pairs of (gradient, variable) where the gradient has been averaged
         across all towers.
      """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
        return average_grads


tf.reset_default_graph()



'''

'''
# tf Graph input
x = tf.placeholder("float", [None, CNN_LSTM_P.n_steps, CNN_LSTM_P.n_input])
y = tf.placeholder("float", [None, n_classes])

traning_file_name_dir, traning_spk_name, spk_name_uniq, labels_one_hot, index_labels = feature_prepare.create_labels(list_file)
features_all = feature_prepare.load_features(traning_file_name_dir, frame_per_second=100)
Segment_valid, labels_valid, labels_train_select, features_traning, index_train_select = feature_prepare.generate_valid_feature(list_file, features_all, CNN_LSTM_P.duration*CNN_LSTM_P.frame_per_second)
features_all = []
BG_CNN_LSTMs_1 = BG_CNN_LSTMs.my_BatchGenerator(features_traning,
                 duration=CNN_LSTM_P.duration, frame_freq = CNN_LSTM_P.frame_per_second, d_frame = CNN_LSTM_P.n_input,
                 batch_size=CNN_LSTM_P.batch_size, labels_train_select = labels_train_select,  index_train_select = index_train_select)


out_CNN_LSTMs = CNN_LSTMs.CNN_MULT_STREAM_LSTMs(x, n_classes, None, None)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = out_CNN_LSTMs.fc3l, labels=y))
optimizer = tf.train.AdamOptimizer().minimize(loss)

correct_pred = tf.equal(tf.argmax(out_CNN_LSTMs.fc3l,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# initial saver
saver = tf.train.Saver()
# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    # batch_x, batch_y = BG_CNN_LSTMs_1.next()

    for num in range(0,CNN_LSTM_P.epoch-1):
        step = 1
        # Segment_train, labels_train = feature_prepare.generate_training_feature_batch(features_traning, labels_train_select, index_train_select)

        while step * CNN_LSTM_P.batch_size < labels_train_select.shape[0]:
            # Run optimization op (backprop)
            # batch_x, batch_y = BG_CNN_LSTMs_1.next()
            # batch_x = Segment_train[(step - 1) * CNN_LSTM_P.batch_size:step * CNN_LSTM_P.batch_size]
            # batch_y = labels_train[(step - 1) * CNN_LSTM_P.batch_size:step * CNN_LSTM_P.batch_size]
            # sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
            batch_x, batch_y = BG_CNN_LSTMs_1.next()
            batch_x_eval = batch_x.eval()
            batch_y_eval = batch_y.eval()
            sess.run(optimizer, feed_dict={x: batch_x_eval, y: batch_y_eval})

            if step % CNN_LSTM_P.display_step == 0:
                # # Calculate batch accuracy
                acc = sess.run(accuracy, feed_dict={x: batch_x_eval, y: batch_y_eval})
                # Calculate batch loss
                loss_value = sess.run(loss, feed_dict={x: batch_x_eval, y: batch_y_eval})
                print("epoch " + str(num+1) + " Iter " + str(step) + ", Minibatch Loss= " + \
                      "{:.3f}".format(loss_value) + ", Training Accuracy= " + \
                      "{:.3f}".format(acc))
                index_v = np.linspace(0, Segment_valid.shape[0] - 1, num=Segment_valid.shape[0])
                np.random.shuffle(index_v)
                index_v = index_v.astype(int)

                acc = sess.run(accuracy, feed_dict={x: Segment_valid[index_v[0:CNN_LSTM_P.batch_size-1]], y: labels_valid[index_v[0:CNN_LSTM_P.batch_size-1]]})
                loss_value = sess.run(loss, feed_dict={x: Segment_valid[index_v[0:CNN_LSTM_P.batch_size-1]], y: labels_valid[index_v[0:CNN_LSTM_P.batch_size-1]]})
                print("epoch " + str(num + 1) + " Iter " + str(step) + ", Minibatch Loss= " + \
                      "{:.3f}".format(loss_value) + ", Validating Accuracy= " + \
                      "{:.3f}".format(acc))
            step += 1


    print("Optimization Finished!")
    save_path = saver.save(sess, "/media/jianboma/Jianbo_Backup1/LSTM_Short_Duration/check/model_CNN_LSTMs.ckpt")
    print("Model saved in file: %s" % save_path)
