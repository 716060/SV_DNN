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
import batch_queue_manage
import gen_batch
import tftables


def list_file_parse(list_file):
    file_name = []
    spk_name = []
    feature_size = []
    with open(list_file) as fp:
        for line in fp:
            line = line.rstrip()  # gets rid of \n (newline)
            #            file, name = line.split(' ')  # line.split returns
            elements = line.split(' ')  # line.split returns
            file_name.append(elements[0])
            spk_name.append(elements[1])
            feature_size.append(elements[2])

    return [file_name, feature_size], spk_name,

#male_TmatrixListfile_100spk
# prepare labels and data    second_layer_training_list_all_mixer training_list_all_less training_validation_list_all_less1 training_list_all_mixer Male_8conv-10sec_testParsed_MFCC_MSR
list_file = '../Trianing_triplet_small_data/evaluation_list/male_TmatrixListfile_100spk_size.scp'
# features, spk_labels = batch_queue_manage.inputs(list_file, CNN_LSTM_P.batcfilename_queue = tf.train.string_input_producer(filenames)h_size)

[filenames, feature_len], spk_names  = list_file_parse(list_file)
feature_len = list(map(int, feature_len))
def filename_queue(filename_list):
    # convert the list to a tensor
    string_tensor = tf.convert_to_tensor(filename_list, dtype=tf.string)
    # randomize the tensor
    # tf.random_shuffle(string_tensor)
    # create the queue
    filename_q = tf.FIFOQueue(capacity=10, dtypes=tf.string)
    # create our enqueue_op for this q
    filename_q_enqueue_op = filename_q.enqueue_many([string_tensor])
    # create a QueueRunner and add to queue runner list
    # we only need one thread for this simple queue
    tf.train.add_queue_runner(tf.train.QueueRunner(filename_q, [filename_q_enqueue_op] * 1))
    return filename_q

def feat_len_queue(feature_len):
    # convert the feature_len to a tensor
    feature_len_tensor = tf.convert_to_tensor(feature_len, dtype=tf.int32)
    # randomize the tensor
    # tf.random_shuffle(feature_len_tensor)
    # create the queue
    feat_len_q = tf.FIFOQueue(capacity=10, dtypes=tf.int32)
    # create our enqueue_op for this q
    feat_len_q_enqueue_op = feat_len_q.enqueue_many([feature_len_tensor])
    # create a QueueRunner and add to queue runner list
    # we only need one thread for this simple queue
    tf.train.add_queue_runner(tf.train.QueueRunner(feat_len_q, [feat_len_q_enqueue_op] * 1))
    return feat_len_q

filename_queue = tf.train.string_input_producer(filenames)
filename = filename_queue.dequeue()
spk_name_queue = tf.train.string_input_producer(spk_names)
spk_name = spk_name_queue.dequeue()
# #
# feature_len_queue = tf.train.string_input_producer(feature_len)
# feat_len = feature_len_queue.dequeue()
# filename_queue = feat_len_queue(feature_len)
feature_len_queue = feat_len_queue(feature_len)
# filename = filename_queue.dequeue()
feat_len = feature_len_queue.dequeue()

reader = tf.TextLineReader()
key, value = reader.read(filename_queue)
record_defaults = [[1.0] for cl in range(39)]

# i0 = tf.constant(0)
# while_body = lambda i, record_format: [i+1, tf.concat([record_format, tf.ones([2, 1], dtype=tf.int32)], axis = 1)]
# while_condition = lambda i, record_format: tf.less(i, feat_len)
#
# record_format_var = tf.while_loop(while_condition, while_body, loop_vars=[i0, record_format],shape_invariants=[i0.get_shape(), tf.TensorShape([2, None])])
#read hdf5 files
loader = tftables.load_dataset(filename='/home/jianboma/Research_Project/myfile.h5',
                                   dataset_path=filename,
                                   input_transform=input_transform,
                                   batch_size=20)

one_frame = tf.stack(tf.decode_csv(
    value, record_defaults=record_defaults))
# one_frame = tf.transpose(one_frame)
# one_frame = tf.expand_dims(one_frame, axis=1)
# i0 = tf.constant(0)
# features = tf.constant([39, 1],dtype=tf.float32)
# while_body = lambda i, features: [i+1, tf.concat([features, one_frame], axis = 1)]
# while_condition = lambda i, record_format: tf.less(i, feat_len)
#
# feature_var = tf.while_loop(while_condition, while_body, loop_vars=[i0, one_frame], shape_invariants=[i0.get_shape(), tf.TensorShape([39, None])]) #shape_invariants=[i0.get_shape(), tf.TensorShape([39, ])]
features = gen_batch.read_multiline([one_frame],
            batch_size=200)
# features = features[1:10]
features_batch, label_batch = tf.train.batch(
            [features, spk_name],
            batch_size=128,
            num_threads=2,
            capacity=1 + 3 * 128)
# feature_var = feature_var[1:-1]
# batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue(
# [features, spk_labels], capacity=2 * CNN_LSTM_P.num_gpus)
# image_batch, label_batch = batch_queue.dequeue()
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    # Start the queue runners.
    threads = tf.train.start_queue_runners(sess=sess)
    print(sess.run(key))
    a = sess.run(e)
    b = sess.run(label_batch)

