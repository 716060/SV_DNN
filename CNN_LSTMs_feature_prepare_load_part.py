

from __future__ import print_function


import h5py
import numpy as np
import CNN_LSTM_Parameters as CNN_LSTM_P



def list_file_parse(list_file):
    file_name = []
    spk_name = []
    with open(list_file) as fp:
        for line in fp:
            line = line.rstrip()  # gets rid of \n (newline)
            elements = line.split(' ')  # line.split returns
            file_name.append(elements[0])
            spk_name.append(elements[1])

    return file_name, spk_name

def get_one_hot(targets, nb_classes):
    return np.eye(nb_classes)[np.array(targets).reshape(-1)]

def create_labels(list_file):
    traning_file_name_dir, traning_spk_name = list_file_parse(list_file)

    spk_name_uniq, spk_name_index, spk_name_index_inv, spk_name_counts = np.unique(traning_spk_name, return_index=True,
                                                                                   return_inverse=True,
                                                                                   return_counts=True)

    index_labels = np.linspace(0, traning_file_name_dir.__len__() - 1, num=traning_file_name_dir.__len__())
    index_labels = index_labels.astype(int)

    return traning_file_name_dir, traning_spk_name, spk_name_uniq, index_labels, spk_name_index_inv


def load_features(file_names1=None, file_names2=None):

    features = []
    f1 = h5py.File('/g/data/wa66/Jianbo/SV_DNN/MFCC_Training_Data.hdf5')

    for file_name in file_names1:
        d1 = f1.get(file_name).value
        features.append(d1)
    return features

def generate_validation_and_training_list(list_file):
    file_name_dir, spk_name, spk_name_uniq, _, spk_name_index_inv = create_labels(list_file)
    index_labels = np.linspace(0, file_name_dir.__len__() - 1, num=file_name_dir.__len__())
    index_labels = index_labels.astype(int)
    # shuffle index
    np.random.shuffle(index_labels)
    #index_labels = index_labels[0:CNN_LSTM_P.folder_file_num]
    # generate validation and training index
    index_valid = index_labels[0:CNN_LSTM_P.valid_file_number]
    index_train_select = index_labels[CNN_LSTM_P.valid_file_number:]
    # generate validation and training file dir and speaker labels

    labels_valid = spk_name_index_inv[index_valid]
    labels_training = spk_name_index_inv[index_train_select]

    file_name_dir = np.asarray(file_name_dir)
    file_name_dir_valid = file_name_dir[index_valid]
    file_name_dir_training = file_name_dir[index_train_select]
    return file_name_dir_valid, labels_valid, file_name_dir_training, labels_training

def generate_training_list_folder(file_name_dir_training, labels_training):

    index_labels = np.linspace(0, file_name_dir_training.__len__() - 1, num=file_name_dir_training.__len__())
    index_labels = index_labels.astype(int)
    # shuffle index
    np.random.shuffle(index_labels)
    index_labels = index_labels[0:CNN_LSTM_P.folder_file_num]
    # generate validation and training file dir and speaker labels

    labels_training = labels_training[index_labels]

    file_name_dir_training = file_name_dir_training[index_labels]
    return file_name_dir_training, labels_training



def generate_feature_all(features_all, labels):

    # feature = load_features(file_name_dir, n_steps=CNN_LSTM_P.n_steps, n_input=CNN_LSTM_P.n_input)
    features = []
    labels_selected = []
    index_labels = np.linspace(0, labels.__len__() - 1, num=labels.__len__())
    index_labels = index_labels.astype(int)
    np.random.shuffle(index_labels)
    for num_gpu in range(0, CNN_LSTM_P.num_gpus):
        for i in range(0, labels.shape[0]):
            d1 = features_all[index_labels[i]]
            if d1.shape[0] > CNN_LSTM_P.n_steps:

                random_start = np.random.randint(0, d1.shape[0] - CNN_LSTM_P.n_steps)
                segment = d1[random_start:random_start + CNN_LSTM_P.n_steps, :]
            else:
                a = np.zeros((CNN_LSTM_P.n_steps - d1.shape[0], CNN_LSTM_P.n_input))
                segment = np.concatenate((d1, a), axis=0)  # padding zeros
            features.append(segment)
            labels_selected.append(labels[index_labels[i]])
    return np.asarray(features), np.asarray(labels_selected)

def generate_feature_valid(features_all, labels):

    # feature = load_features(file_name_dir, n_steps=CNN_LSTM_P.n_steps, n_input=CNN_LSTM_P.n_input)
    features = []
    labels_selected = []
    index_labels = np.linspace(0, labels.__len__() - 1, num=labels.__len__())
    index_labels = index_labels.astype(int)

    for num_gpu in range(0, CNN_LSTM_P.num_gpus):
        np.random.shuffle(index_labels)
        for i in range(0, CNN_LSTM_P.batch_size):
            d1 = features_all[index_labels[i]]
            if d1.shape[0] > CNN_LSTM_P.n_steps:

                random_start = np.random.randint(0, d1.shape[0] - CNN_LSTM_P.n_steps)
                segment = d1[random_start:random_start + CNN_LSTM_P.n_steps, :]
            else:
                a = np.zeros((CNN_LSTM_P.n_steps - d1.shape[0], CNN_LSTM_P.n_input))
                segment = np.concatenate((d1, a), axis=0)  # padding zeros
            features.append(segment)
            labels_selected.append(labels[index_labels[i]])
    return np.asarray(features), np.asarray(labels_selected)

def validation_data(features_valid, labels_valid):
    return generate_feature_valid(features_valid, labels_valid)

def training_data(features_training, labels_training):
    return generate_feature_all(features_training, labels_training)


