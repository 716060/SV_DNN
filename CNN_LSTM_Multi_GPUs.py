
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

    traning_file_name_dir, traning_spk_name, spk_name_uniq, labels_one_hot, index_labels = feature_prepare.create_labels(list_file)
    features_all = feature_prepare.load_features(traning_file_name_dir, frame_per_second=100)
    Segment_valid, labels_valid, labels_train_select, features_traning, index_train_select = feature_prepare.generate_valid_feature(list_file, features_all)
    features_all = []
    BG_CNN_LSTMs_1 = BG_CNN_LSTMs.my_BatchGenerator(features_traning,
                 duration=CNN_LSTM_P.duration, frame_freq = CNN_LSTM_P.frame_per_second, d_frame = CNN_LSTM_P.n_input,
                 batch_size=CNN_LSTM_P.batch_size, labels_train_select = labels_train_select,  index_train_select = index_train_select)
    batch_x, batch_y = BG_CNN_LSTMs_1.batch_generate()

    for num in range(0,CNN_LSTM_P.epoch-1):
        step = 1
        Segment_train, labels_train = feature_prepare.generate_training_feature_batch(features_traning, labels_train_select, index_train_select)

        while step * CNN_LSTM_P.batch_size < Segment_train.shape[0]:
            # Run optimization op (backprop)
            batch_x = Segment_train[(step - 1) * CNN_LSTM_P.batch_size:step * CNN_LSTM_P.batch_size]
            batch_y = labels_train[(step - 1) * CNN_LSTM_P.batch_size:step * CNN_LSTM_P.batch_size]
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})

            if step % CNN_LSTM_P.display_step == 0:
                # Calculate batch accuracy
                acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
                # Calculate batch loss
                loss_value = sess.run(loss, feed_dict={x: batch_x, y: batch_y})
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



def train():
    with tf.Graph().as_default():
        with tf.device('/cpu:0'):
            # Create a variable to count the number of train() calls. This equals the
            # number of batches processed * FLAGS.num_gpus.
            global_step = tf.get_variable(
                'global_step', [],
                initializer=tf.constant_initializer(0), trainable=False)

            # Calculate the learning rate schedule.
            # num_batches_per_epoch = (CNN_LSTM_P.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN /
            #                          CNN_LSTM_P.batch_size)
            # decay_steps = int(num_batches_per_epoch * CNN_LSTM_P.NUM_EPOCHS_PER_DECAY)
            #
            # # Decay the learning rate exponentially based on the number of steps.
            # lr = tf.train.exponential_decay(CNN_LSTM_P.INITIAL_LEARNING_RATE,
            #                                 global_step,
            #                                 decay_steps,
            #                                 CNN_LSTM_P.LEARNING_RATE_DECAY_FACTOR,
            #                                 staircase=True)

            # Create an optimizer that performs gradient descent.
            opt = tf.train.GradientDescentOptimizer(CNN_LSTM_P.learning_rate)

            BG_CNN_LSTMs_1 = BG_CNN_LSTMs.my_BatchGenerator(features_traning,
                                                            duration=CNN_LSTM_P.duration,
                                                            frame_freq=CNN_LSTM_P.frame_per_second,
                                                            d_frame=CNN_LSTM_P.n_input,
                                                            batch_size=CNN_LSTM_P.batch_size)
            # Get images and labels for CIFAR-10.
            # images, labels = CNN_LSTM_P.distorted_inputs()
            # batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue(
            #     [images, labels], capacity=2 * CNN_LSTM_P.num_gpus)
            # Calculate the gradients for each model tower.
            tower_grads = []
            with tf.variable_scope(tf.get_variable_scope()):
                for i in xrange(CNN_LSTM_P.num_gpus):
                    with tf.device('/gpu:%d' % i):
                        with tf.name_scope('%s_%d' % (CNN_LSTM_P.TOWER_NAME, i)) as scope:
                            # Dequeues one batch for the GPU
                            # image_batch, label_batch = batch_queue.dequeue()
                            batch_x, batch_y = BG_CNN_LSTMs_1.batch_generate()
                            # Calculate the loss for one tower of the CIFAR model. This function
                            # constructs the entire CIFAR model but shares the variables across
                            # all towers.
                            loss = tower_loss(scope, batch_x, batch_y)

                            # Reuse variables for the next tower.
                            tf.get_variable_scope().reuse_variables()

                            # Retain the summaries from the final tower.
                            summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

                            # Calculate the gradients for the batch of data on this CIFAR tower.
                            grads = opt.compute_gradients(loss)

                            # Keep track of the gradients across all towers.
                            tower_grads.append(grads)

            # We must calculate the mean of each gradient. Note that this is the
            # synchronization point across all towers.
            grads = average_gradients(tower_grads)

            # Add a summary to track the learning rate.
            summaries.append(tf.summary.scalar('learning_rate', CNN_LSTM_P.learning_rate))

            # Add histograms for gradients.
            for grad, var in grads:
                if grad is not None:
                    summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))

            # Apply the gradients to adjust the shared variables.
            apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

            # Add histograms for trainable variables.
            for var in tf.trainable_variables():
                summaries.append(tf.summary.histogram(var.op.name, var))

            # Track the moving averages of all trainable variables.
            variable_averages = tf.train.ExponentialMovingAverage(
                CNN_LSTM_P.MOVING_AVERAGE_DECAY, global_step)
            variables_averages_op = variable_averages.apply(tf.trainable_variables())

            # Group all updates to into a single train op.
            train_op = tf.group(apply_gradient_op, variables_averages_op)

            # Create a saver.
            saver = tf.train.Saver(tf.global_variables())

            # Build the summary operation from the last tower summaries.
            summary_op = tf.summary.merge(summaries)

            # Build an initialization operation to run below.
            init = tf.global_variables_initializer()

            # Start running operations on the Graph. allow_soft_placement must be set to
            # True to build towers on GPU, as some of the ops do not have GPU
            # implementations.
            sess = tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=CNN_LSTM_P.log_device_placement))
            sess.run(init)

            # Start the queue runners.
            tf.train.start_queue_runners(sess=sess)

            summary_writer = tf.summary.FileWriter(CNN_LSTM_P.train_dir, sess.graph)

            for step in xrange(CNN_LSTM_P.epoch):
                start_time = time.time()
                _, loss_value = sess.run([train_op, loss])
                duration = time.time() - start_time

                assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

                if step % 10 == 0:
                    num_examples_per_step = CNN_LSTM_P.batch_size * CNN_LSTM_P.num_gpus
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = duration / CNN_LSTM_P.num_gpus

                    format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                                  'sec/batch)')
                    print(format_str % (datetime.now(), step, loss_value,
                                        examples_per_sec, sec_per_batch))

                if step % 100 == 0:
                    summary_str = sess.run(summary_op)
                    summary_writer.add_summary(summary_str, step)

                # Save the model checkpoint periodically.
                if step % 1000 == 0 or (step + 1) == CNN_LSTM_P.epoch:
                    checkpoint_path = os.path.join(CNN_LSTM_P.train_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)
                    if step % 1000 == 0 or (step + 1) == CNN_LSTM_P.max_steps:
                        checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                        saver.save(sess, checkpoint_path, global_step=step)






def main(argv=None):  # pylint: disable=unused-argument
    CNN_LSTM_P.maybe_download_and_extract()
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    train()



if __name__ == '__main__':
    tf.app.run()



# def train():
#     with tf.Graph().as_default():
#         with tf.device('/cpu:0'):
#             # Create a variable to count the number of train() calls. This equals the
#             # number of batches processed * FLAGS.num_gpus.
#             global_step = tf.get_variable(
#                 'global_step', [],
#                 initializer=tf.constant_initializer(0), trainable=False)
#
#             # Calculate the learning rate schedule.
#             num_batches_per_epoch = (CNN_LSTM_P.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN /
#                                      FLAGS.batch_size)
#             decay_steps = int(num_batches_per_epoch * CNN_LSTM_P.NUM_EPOCHS_PER_DECAY)
#
#             # Decay the learning rate exponentially based on the number of steps.
#             lr = tf.train.exponential_decay(CNN_LSTM_P.INITIAL_LEARNING_RATE,
#                                             global_step,
#                                             decay_steps,
#                                             CNN_LSTM_P.LEARNING_RATE_DECAY_FACTOR,
#                                             staircase=True)
#
#             # Create an optimizer that performs gradient descent.
#             opt = tf.train.GradientDescentOptimizer(lr)
#
#             # Get images and labels for CIFAR-10.
#             images, labels = CNN_LSTM_P.distorted_inputs()
#             batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue(
#                 [images, labels], capacity=2 * FLAGS.num_gpus)
#             # Calculate the gradients for each model tower.
#             tower_grads = []
#             with tf.variable_scope(tf.get_variable_scope()):
#                 for i in xrange(FLAGS.num_gpus):
#                     with tf.device('/gpu:%d' % i):
#                         with tf.name_scope('%s_%d' % (CNN_LSTM_P.TOWER_NAME, i)) as scope:
#                             # Dequeues one batch for the GPU
#                             image_batch, label_batch = batch_queue.dequeue()
#                             # Calculate the loss for one tower of the CIFAR model. This function
#                             # constructs the entire CIFAR model but shares the variables across
#                             # all towers.
#                             loss = tower_loss(scope, image_batch, label_batch)
#
#                             # Reuse variables for the next tower.
#                             tf.get_variable_scope().reuse_variables()
#
#                             # Retain the summaries from the final tower.
#                             summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
#
#                             # Calculate the gradients for the batch of data on this CIFAR tower.
#                             grads = opt.compute_gradients(loss)
#
#                             # Keep track of the gradients across all towers.
#                             tower_grads.append(grads)
#
#             # We must calculate the mean of each gradient. Note that this is the
#             # synchronization point across all towers.
#             grads = average_gradients(tower_grads)
#
#             # Add a summary to track the learning rate.
#             summaries.append(tf.summary.scalar('learning_rate', lr))
#
#             # Add histograms for gradients.
#             for grad, var in grads:
#                 if grad is not None:
#                     summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))
#
#             # Apply the gradients to adjust the shared variables.
#             apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
#
#             # Add histograms for trainable variables.
#             for var in tf.trainable_variables():
#                 summaries.append(tf.summary.histogram(var.op.name, var))
#
#             # Track the moving averages of all trainable variables.
#             variable_averages = tf.train.ExponentialMovingAverage(
#                 CNN_LSTM_P.MOVING_AVERAGE_DECAY, global_step)
#             variables_averages_op = variable_averages.apply(tf.trainable_variables())
#
#             # Group all updates to into a single train op.
#             train_op = tf.group(apply_gradient_op, variables_averages_op)
#
#             # Create a saver.
#             saver = tf.train.Saver(tf.global_variables())
#
#             # Build the summary operation from the last tower summaries.
#             summary_op = tf.summary.merge(summaries)
#
#             # Build an initialization operation to run below.
#             init = tf.global_variables_initializer()
#
#             # Start running operations on the Graph. allow_soft_placement must be set to
#             # True to build towers on GPU, as some of the ops do not have GPU
#             # implementations.
#             sess = tf.Session(config=tf.ConfigProto(
#                 allow_soft_placement=True,
#                 log_device_placement=FLAGS.log_device_placement))
#             sess.run(init)
#
#             # Start the queue runners.
#             tf.train.start_queue_runners(sess=sess)
#
#             summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)
#
#             for step in xrange(FLAGS.max_steps):
#                 start_time = time.time()
#                 _, loss_value = sess.run([train_op, loss])
#                 duration = time.time() - start_time
#
#                 assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
#
#                 if step % 10 == 0:
#                     num_examples_per_step = FLAGS.batch_size * FLAGS.num_gpus
#                     examples_per_sec = num_examples_per_step / duration
#                     sec_per_batch = duration / FLAGS.num_gpus
#
#                     format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
#                                   'sec/batch)')
#                     print(format_str % (datetime.now(), step, loss_value,
#                                         examples_per_sec, sec_per_batch))
#
#                 if step % 100 == 0:
#                     summary_str = sess.run(summary_op)
#                     summary_writer.add_summary(summary_str, step)
#
#                 # Save the model checkpoint periodically.
#                 if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
#                     checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
#                     saver.save(sess, checkpoint_path, global_step=step)
#                     if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
#                         checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
#                         saver.save(sess, checkpoint_path, global_step=step)






