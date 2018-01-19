

import numpy as np
import h5py

import tensorflow as tf
from tensorflow.contrib import rnn
import sidekit
import list_file_parse as LFP


# prepare labels and data    second_layer_training_list_all_mixer training_list_all_less training_validation_list_all_less1 training_list_all_mixer Male_8conv-10sec_testParsed_MFCC_MSR
list_file = '../Trianing_triplet_small_data/evaluation_list/training_list_all_mixer.scp'

traning_file_name_dir, traning_spk_name = LFP.list_file_parse(list_file)



frame_per_second = 100

file = h5py.File("training_mixer_male.hdf5", "w")

for filename, spk_name  in [traning_file_name_dir, traning_spk_name]:
    features, fp, dt, tc, t = sidekit.frontend.io.read_htk(filename, frame_per_second=frame_per_second)
    elements = filename.split('/')
    dataset = file.create_dataset(elements[-1], features.shape, features.dtype, id=spk_name)
    dataset[...] = features


##obtain data
# filename = 'training_mixer_male.hdf5'
# f = h5py.File(filename, 'r')
# a_group_key = list(f.keys())[0]
# data = list(f[a_group_key])
