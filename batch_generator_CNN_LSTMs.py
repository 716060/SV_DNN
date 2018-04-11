

import warnings
import list_file_parse as LFP
import sidekit
import itertools
import tensorflow as tf
from scipy.spatial.distance import pdist, squareform
from pyannote.generators.batch import BaseBatchGenerator
from pyannote.audio.generators.labels import \
    LabeledFixedDurationSequencesBatchGenerator
import threading




class my_BatchGenerator(object):
    """Batch generator for CNN_LSTMs embedding
    Generates ([Xa, Xp, Xn], 1) tuples where
      * Xa is the anchor sequence (e.g. by speaker S)
      * Xp is the positive sequence (also uttered by speaker S)
      * Xn is the negative sequence (uttered by a different speaker)
    and such that d(f(Xa), f(Xn)) < d(f(Xa), f(Xp)) + margin where
      * f is the current state of the embedding network (being optimized)
      * d is the euclidean distance
      * margin is the triplet loss margin (e.g. 0.2, typically)
    Parameters
    ----------
    extractor: YaafeFeatureExtractor
        Yaafe feature extraction (e.g. YaafeMFCC instance)
    file_generator: iterable
        File generator (the training set, typically)
    embedding: SequenceEmbedding
        Sequence embedding (currently being optimized)
    duration: float, optional
        Sequence duration. Defaults to 3 seconds.
    overlap: float, optional
        Sequence overlap ratio. Defaults to 0 (no overlap).
    normalize: boolean, optional
        When True, normalize sequence (z-score). Defaults to False.
    per_label: int, optional
        Number of samples per label. Defaults to 40.
    per_fold: int, optional
        When provided, randomly split the training set into
        fold of `per_fold` labels (e.g. 40) after each epoch.
        Defaults to using the whole traning set.
    batch_size: int, optional
        Batch size. Defaults to 32.
    """

    def __init__(self, features,
                 duration=2.0, frame_freq = 100, d_frame = 39,
                 batch_size=128, labels_train_select = None,  index_train_select = None):

        # super(batch_generator_CNN_LSTMs, self).__init__()


        self.features = features
        self.duration = duration
        self.frame_freq = frame_freq
        self.n_step = self.duration*self.frame_freq
        self.d_frame = d_frame
        self.batch_size = batch_size
        self.labels_train_select = labels_train_select
        self.index_train_select = index_train_select
        self.lock = threading.Lock()
        self.batch_generate_ = self.batch_generate()
        self.generate_training_samples_from_loaded_features()

        # consume first element of generator
        # this is meant to pre-generate all labeled sequences once and for all
        # and get the number of unique labels into self.n_labels
        next(self.batch_generate_)


    def batch_generate(self):

        # infinite loop
        while True:
            index_train = tf.lin_space(0.0, self.index_train_select.__len__() - 1, num=self.index_train_select.__len__())
            segmentclass = self.generate_training_samples_from_loaded_features(self.features, self.duration*self.frame_freq)
            Segment_train = []
            counter = 0
            for segment in segmentclass:
                Segment_train.append(segment)
                # print(counter)
                counter += 1
            # Segment_train = tf.asarray(Segment_train)

            # shuffle training data
            tf.random_shuffle(index_train)
            index_train = index_train.astype(int)
            labels_train = tf.gather_nd(self.labels_train_select, index_train)
            Segment_train = tf.gather_nd(Segment_train, index_train)
            yield Segment_train, labels_train


    def generate_training_samples_from_loaded_features(self, n_steps=200, n_input=39):

        for d1 in self.features:
            if d1.shape[0] > n_steps:

                random_start = tf.random_uniform(1, minval= 0, maxval = d1.shape[0] - n_steps, dtype = tf.int32)
                segment = d1[random_start:random_start + n_steps, :]
            else:
                a = tf.zeros((n_steps - d1.shape[0], n_input))
                segment = tf.concat([d1, a], axis=0)  # padding zeros

            yield segment



    def __iter__(self):
        return self


    def next(self):
        return self.__next__()


    def __next__(self):
        return next(self.batch_generate())



    def get_shape(self):
        return (self.duration*self.frame_freq, self.d_frame)

    def signature(self):
        shape = self.get_shape()
        return (
            [
                {'type': 'sequence', 'shape': shape},
                {'type': 'sequence', 'shape': shape},
                {'type': 'sequence', 'shape': shape}
            ],
            {'type': 'boolean'}
        )







