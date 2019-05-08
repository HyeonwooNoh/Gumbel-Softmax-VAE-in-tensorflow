import os
import numpy as np
import tensorflow as tf

from util.preprocess import ZCA, ContrastNorm


class MNISTLoader(object):
    '''
    I assume that 'download' and 'unzip' was done outside

    Training set: 60k
    Test set: 10k

    Subscript:
        u: unlabeled
        l: labeled
        t: test
    '''
    def __init__(self, path, config=None):
        if config is None:
            self.config = {'name': 'mnist'}
        else:
            self.config = config

        self.x_l = self.load_images(
            filename=os.path.join(path, 'train-images-idx3-ubyte'),
            N=60000)
        self.y_l = self.load_labels(
            filename=os.path.join(path, 'train-labels-idx1-ubyte'),
            N=60000)
        self.x_t = self.load_images(
            filename=os.path.join(path, 't10k-images-idx3-ubyte'),
            N=10000)
        self.y_t = self.load_labels(
            filename=os.path.join(path, 't10k-labels-idx1-ubyte'),
            N=10000)
        self.x_u = None
        self.y_u = None  # [TODO] actually shouldn't exist

    def load_images(self, filename, N):
        '''
        Load MNIST from the downloaded and unzipped file
        into [N, 28, 28, 1] dimensions with [0, 1] values
        '''
        with open(filename) as f:
            x = np.fromfile(file=f, dtype=np.uint8)
        x = x[16:].reshape([N, 28, 28, 1]).astype(np.float32)
        x = x / 255.
        return x

    def load_labels(self, filename, N):
        ''' `int` '''
        with open(filename) as f:
            x = np.fromfile(file=f, dtype=np.uint8)
        x = x[8:].reshape([N]).astype(np.int32)
        return x

    # ==== Ad-hoc: the following are for SSL only ====
    def divide_semisupervised(self, N_u):
        '''
        Shuffle before splitting
        '''
        idx = np.arange(self.y_l.shape[0])
        self.x_l = self.x_l[idx]
        self.y_l = self.y_l[idx]

        self.x_u = self.x_l[:N_u]
        self.y_u = self.y_l[:N_u]

        self.x_l = self.x_l[N_u:]
        self.y_l = self.y_l[N_u:]

    def pick_supervised_samples(self, smp_per_class=10):
        ''' [TODO] improve it '''
        idx = np.arange(self.y_l.shape[0])
        rng_state = np.random.RandomState(seed=123)
        rng_state.shuffle(idx)
        x = self.x_l[idx]
        y = self.y_l[idx]
        index = list()
        for i in range(10):
            count = 0
            index_ = list()
            for j in range(y.shape[0]):
                if y[j] == i and count < smp_per_class:
                    index_.append(j)
                    count += 1
            index = index + index_
        # x_s = x[index]
        # y_s = y[index]
        return x[index], y[index]
    # =======================================


class CIFAR10Loader(object):
    '''
    Subscript:
        u: unlabeled
        l: labeled
        t: test
    '''
    def __init__(self, path, config=None):
        if config is None:
            self.config = {'name': 'cifar',
                           'contrast_norm': {'scale': 55},
                           'whiten_zca': {'n_components': 3072}}
        else:
            self.config = config

        self.path = path
        (x_l, y_l), (x_t, y_t) = tf.keras.datasets.cifar10.load_data()
        self.x_l = x_l.astype(np.float32)
        self.y_l = np.squeeze(y_l)
        self.x_t = x_t.astype(np.float32)
        self.y_t = np.squeeze(y_t)
        self.x_u = None
        self.y_u = None  # [TODO] actually shouldn't exist
        self.rng_state = np.random.RandomState(seed=234)

        self.normalize_data()

    def normalize_data(self):
        if 'centering' in self.config:
            self.x_l = self.x_l / 128.0 - 1.0
            self.x_t = self.x_t / 128.0 - 1.0
        if 'contrast_norm' in self.config:
            cnorm = ContrastNorm(self.config['contrast_norm']['scale'])
            self.x_l = cnorm.apply(self.x_l)
            self.x_t = cnorm.apply(self.x_t)
        if 'whiten_zca' in self.config:
            whiten = ZCA()
            whiten.fit(self.config['whiten_zca']['n_components'], self.x_l)
            self.x_l = whiten.apply(self.x_l)
            self.x_t = whiten.apply(self.x_t)

    # ==== Ad-hoc: the following are for SSL only ====
    def divide_semisupervised(self, N_u):
        '''
        Shuffle before splitting
        '''
        idx = np.arange(self.y_l.shape[0])
        self.rng_state.shuffle(idx)
        self.x_l = self.x_l[idx]
        self.y_l = self.y_l[idx]

        self.x_u = self.x_l
        self.y_u = self.y_l

    def pick_supervised_samples(self, smp_per_class=10):
        ''' [TODO] improve it '''
        idx = np.arange(self.y_l.shape[0])
        self.rng_state.shuffle(idx)
        x = self.x_l[idx]
        y = self.y_l[idx]
        index = list()
        for i in range(10):
            count = 0
            index_ = list()
            for j in range(y.shape[0]):
                if y[j] == i and count < smp_per_class:
                    index_.append(j)
                    count += 1
            index = index + index_
        return x[index], y[index]
