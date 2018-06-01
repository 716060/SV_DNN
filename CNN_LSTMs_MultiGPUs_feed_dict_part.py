
'''
a CNN with a stream of Bi-LSTMs are used to produce a embedding for speaker verification
Author: Jianbo Ma
Project:
'''

from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import time
import re
import os
import gc
import sys
from datetime import datetime
import CNN_LSTMs_Model_Variable_Scope as CNN_LSTMs
import CNN_LSTMs_feature_prepare_load_part as CNN_LSTMs_feature_prepare
import CNN_LSTM_Parameters as CNN_LSTM_P

# define the stdout of this project
#orig_stdout = sys.stdout
#output_file = '/home/561/jm6639/log_file/log_CNN_LSTMs.txt'
#f = open(output_file, 'w')
#sys.stdout = f

f = open('/home/561/jm6639/log_file/training_validate_acc.txt','w+')
list_file = '/home/561/jm6639/python_script/Multi_GPUs/List_File/DL_training_filename.scp'
_, _, spk_name_uniq,_,_ = CNN_LSTMs_feature_prepare.create_labels(list_file)
n_classes = spk_name_uniq.__len__()

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


# def tower_loss(scope, input_features, labels, n_classes):
def tower_loss(scope, input_features, labels, n_classes, lstm_fw_cell, lstm_bw_cell):

    """Calculate the total loss on a single tower running the CIFAR model.
      Args:
    	scope: unique prefix string identifying the CIFAR tower, e.g. 'tower_0'
    	images: Images. 4D tensor of shape [batch_size, height, width, 3].
    	labels: Labels. 1D tensor of shape [batch_size].
      Returns:
    	 Tensor of shape [] containing the total loss for a batch of data
      """

    # Build inference Graph.
    # Build the portion of the Graph calculating the losses. Note that we will
    # assemble the total_loss using a custom function below.
    logits = CNN_LSTMs.Model(input_features, n_classes, lstm_fw_cell, lstm_bw_cell)
    # logits = CNN_LSTMs.Model(input_features, n_classes)
    _ = CNN_LSTMs.loss(logits, labels)

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

    # training accuracy
    correct_pred = tf.equal(tf.argmax(logits, 1), labels)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    return total_loss, accuracy



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

# def validation_acc_loss(input_features, labels, n_classes, lstm_fw_cell, lstm_bw_cell):
#     """Calculate the total loss on a single tower running the CIFAR model.
#       Args:
#     	scope: unique prefix string identifying the CIFAR tower, e.g. 'tower_0'
#     	images: Images. 4D tensor of shape [batch_size, height, width, 3].
#     	labels: Labels. 1D tensor of shape [batch_size].
#       Returns:
#     	 Tensor of shape [] containing the total loss for a batch of data
#       """
#
#     # Build inference Graph.
#     # Build the portion of the Graph calculating the losses. Note that we will
#     # assemble the total_loss using a custom function below.
#     logits = CNN_LSTMs.Model(input_features, n_classes, lstm_fw_cell, lstm_bw_cell)
#     _ = CNN_LSTMs.loss(logits, labels)
#
#     # Assemble all of the losses for the current tower only.
#     losses = tf.get_collection('losses', scope)
#
#     # Calculate the total loss for the current tower.
#     total_loss = tf.add_n(losses, name='total_loss')
#
#     # Attach a scalar summary to all individual losses and the total loss; do the
#     # same for the averaged version of the losses.
#     for l in losses + [total_loss]:
#         # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
#         # session. This helps the clarity of presentation on tensorboard.
#         loss_name = re.sub('%s_[0-9]*/' % CNN_LSTM_P.TOWER_NAME, '', l.op.name)
#         tf.summary.scalar(loss_name, l)
#
#     # training accuracy
#     correct_pred = tf.equal(tf.argmax(logits, 1), labels)
#     accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
#
#     return total_loss, accuracy
#
file_name_dir_valid, labels_valid, file_name_dir_training_all, labels_training_all = CNN_LSTMs_feature_prepare.generate_validation_and_training_list(list_file)
features_validation_all = CNN_LSTMs_feature_prepare.load_features(file_name_dir_valid)

tf.reset_default_graph()
'''

'''
# tf Graph input
with tf.Graph().as_default():#, tf.device('/cpu:0')
    # with tf.device('/cpu:0'):
    # Create a variable to count the number of train() calls. This equals the
    # number of batches processed * FLAGS.num_gpus.

    # define inputs
    x = tf.placeholder("float", [CNN_LSTM_P.num_gpus, None, CNN_LSTM_P.n_steps, CNN_LSTM_P.n_input])
    y = tf.placeholder("int64", [CNN_LSTM_P.num_gpus, None])

     # define inputs for validation
    # x_validation = tf.placeholder("float", [None, CNN_LSTM_P.n_steps, CNN_LSTM_P.n_input])
    # y_validation = tf.placeholder("int64", [None])

    lstm_fw_cell = rnn.BasicLSTMCell(128, forget_bias=1.0)
    lstm_bw_cell = rnn.BasicLSTMCell(128, forget_bias=1.0)

    global_step = tf.get_variable(
        'global_step', [],
        initializer=tf.constant_initializer(0), trainable=False)

    # Create an optimizer that performs gradient descent.
    # opt = tf.train.GradientDescentOptimizer(CNN_LSTM_P.learning_rate)
    opt = tf.train.AdamOptimizer(CNN_LSTM_P.learning_rate)



    # Calculate the gradients for each model tower.
    tower_grads = []
    acc_all = []
    with tf.variable_scope(tf.get_variable_scope()):
        for i in range(0, CNN_LSTM_P.num_gpus):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('%s_%d' % (CNN_LSTM_P.TOWER_NAME, i)) as scope:
                    # Dequeues one batch for the GPU
                    # feature_batch, label_batch = batch_queue.dequeue()



                    # Calculate the loss for one tower of the CIFAR model. This function
                    # constructs the entire CIFAR model but shares the variables across
                    # all towers.
                    loss, acc = tower_loss(scope, x[i],
                                      y[i], n_classes, lstm_fw_cell, lstm_bw_cell)
                    # loss, acc = tower_loss(scope, x[i],
                    #                        y[i], n_classes)
                    # add acc for each gpu
                    acc_all.append(acc)
                    # Reuse variables for the next tower.
                    tf.get_variable_scope().reuse_variables()

                    # Retain the summaries from the final tower.
                    summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

                    # Calculate the gradients for the batch of data on this CIFAR tower.
                    grads = opt.compute_gradients(loss)

                    # Keep track of the gradients across all towers.
                    tower_grads.append(grads)

                    # directly using loss function
                    # optimizer = tf.train.AdamOptimizer(learning_rate=CNN_LSTM_P.learning_rate).minimize(loss)

    acc_all = tf.reduce_mean(acc_all)
    # We must calculate the mean of each gradient. Note that this is the
    # synchronization point across all towers.
    grads = average_gradients(tower_grads)
    # grads = tower_grads[0]
    # # calculate accuracy
    # acc_all = tf.reduce_mean(acc_all)
    # # Add a summary to track the learning rate.
    # summaries.append(tf.summary.scalar('learning_rate', CNN_LSTM_P.learning_rate))
    #
    # # Add histograms for gradients.
    # for grad, var in grads:
    #     if grad is not None:
    #         summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))
    #
    # Apply the gradients to adjust the shared variables.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
    #
    # # Add histograms for trainable variables.
    # for var in tf.trainable_variables():
    #     summaries.append(tf.summary.histogram(var.op.name, var))

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        CNN_LSTM_P.MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    # #
    # Group all updates to into a single train op.
    train_op = tf.group(apply_gradient_op, variables_averages_op)

    # add validation loss and acc
    # with tf.device('/gpu:%d' % i):
    #     loss, acc = tower_loss(x_validation,y_validation, n_classes, lstm_fw_cell, lstm_bw_cell)

    # Create a saver.
    saver = tf.train.Saver(tf.global_variables())
    #
    # # Build the summary operation from the last tower summaries.
    # summary_op = tf.summary.merge(summaries)

    # Build an initialization operation to run below.
    init = tf.global_variables_initializer()

    # Start running operations on the Graph. allow_soft_placement must be set to
    # True to build towers on GPU, as some of the ops do not have GPU
    # implementations.
    sess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=CNN_LSTM_P.log_device_placement))
    sess.run(init)
    #load trained model
    saver.restore(sess, "/g/data/wa66/Jianbo/SV_DNN/model_store/model_part.ckpt-175")
   # Start the queue runners.
    tf.train.start_queue_runners(sess=sess)

    summary_writer = tf.summary.FileWriter(CNN_LSTM_P.train_dir, sess.graph)

    for step in range(230, CNN_LSTM_P.pseudo_epoch):
        folder_iter = 0
        if step > 230:
            del features_training_all
            gc.collect()
        file_name_dir_training, labels_training = CNN_LSTMs_feature_prepare.generate_training_list_folder(
            file_name_dir_training_all,
            labels_training_all)
        features_training_all = CNN_LSTMs_feature_prepare.load_features(
            file_name_dir_training) 
        while folder_iter < CNN_LSTM_P.folder_max_steps:
            folder_iter += 1
            start_time = time.time()

            training_data_batches, training_labels_batches = CNN_LSTMs_feature_prepare.training_data(features_training=features_training_all,
                                                                                                     labels_training=labels_training)

            # training_data_batches, training_labels_batches = CNN_LSTMs_feature_prepare.training_data(
            #     features_training=features_training_all,
            #     labels_training=labels_training)

            count = 0
            while(count+CNN_LSTM_P.num_gpus*CNN_LSTM_P.batch_size<training_data_batches.__len__()):
                training_data_batch = []
                training_labels_batch = []
                for i in range(0, CNN_LSTM_P.num_gpus):
                    training_data_batch.append(training_data_batches[count:count + CNN_LSTM_P.batch_size,:,:])
                    training_labels_batch.append(training_labels_batches[count:count + CNN_LSTM_P.batch_size])
                    count += CNN_LSTM_P.batch_size
                
                training_data_batch = np.asarray(training_data_batch)
                #print(training_data_batch.shape)
                training_data_batch = np.reshape(training_data_batch,
                                                 newshape=[CNN_LSTM_P.num_gpus, CNN_LSTM_P.batch_size, CNN_LSTM_P.n_steps,
                                                           CNN_LSTM_P.n_input])
                training_labels_batch = np.reshape(training_labels_batch,
                                                   newshape=[CNN_LSTM_P.num_gpus, CNN_LSTM_P.batch_size])

                _, loss_value, acc = sess.run([train_op, loss, acc_all], feed_dict={x: training_data_batch, y: training_labels_batch})
                # _, loss_value = sess.run([train_op, loss]) #optimizer
                duration = time.time() - start_time

                assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

                if count % 20*CNN_LSTM_P.batch_size == 0:
                    num_examples_per_step = CNN_LSTM_P.batch_size * CNN_LSTM_P.num_gpus
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = duration / CNN_LSTM_P.num_gpus
                    # validation
                    validation_data_batches, validation_labels_batches = CNN_LSTMs_feature_prepare.validation_data(
                        features_valid=features_validation_all,labels_valid=labels_valid)
                    validation_data_batches = np.reshape(validation_data_batches,
                                                     newshape=[CNN_LSTM_P.num_gpus, CNN_LSTM_P.batch_size,
                                                               CNN_LSTM_P.n_steps,
                                                               CNN_LSTM_P.n_input])
                    validation_data_batches = np.asarray(validation_data_batches)
                    validation_labels_batches = np.reshape(validation_labels_batches,
                                                       newshape=[CNN_LSTM_P.num_gpus, CNN_LSTM_P.batch_size])
                    acc_vallidation = sess.run(acc_all, feed_dict={x: validation_data_batches, y: validation_labels_batches})
                    format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                                  'sec/batch) trainng_acc = %.3f, validation_acc = %.3f')
                    f.write(format_str % (datetime.now(), step, loss_value,
                                        examples_per_sec, sec_per_batch, acc, acc_vallidation))
                    print(format_str % (datetime.now(), step, loss_value,
                                        examples_per_sec, sec_per_batch, acc, acc_vallidation))

        # if step % 100 == 0:
        #     summary_str = sess.run(summary_op, feed_dict={x: training_data_batches, y: training_labels_batches})
        #     summary_writer.add_summary(summary_str, step)

            # Save the model checkpoint periodically.
            if step % 5 == 0 or (step + 1) == CNN_LSTM_P.pseudo_epoch:
                checkpoint_path = os.path.join(CNN_LSTM_P.train_dir, 'model_part.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
                if step % 5 == 0 or (step + 1) == CNN_LSTM_P.max_steps:
                    checkpoint_path = os.path.join(CNN_LSTM_P.train_dir, 'model_part.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)


# def main(argv=None):  # pylint: disable=unused-argument
#     CNN_LSTM_P.maybe_download_and_extract()
#     if tf.gfile.Exists(FLAGS.train_dir):
#         tf.gfile.DeleteRecursively(FLAGS.train_dir)
#     tf.gfile.MakeDirs(FLAGS.train_dir)
#     train()
#
#
#
# if __name__ == '__main__':
#     tf.app.run()

#sys.stdout = orig_stdout
#f.close()
f.close()
