import sys
import os
import gzip
import numpy as np
import pickle
import scipy.io as sio


def to_one_hot(x, depth):
    ret = np.zeros((x.shape[0], depth), dtype=np.int32)
    ret[np.arange(x.shape[0]), x] = 1
    return ret


def get_ssl_data(x_train, y_train, num_labeled, n_y, seed=None, save_dir=None):
    if num_labeled is None:
        return x_train, y_train
    if seed is None or seed < 0:
        seed = 1234
    else:
        seed = int(seed)
    rng_data = np.random.RandomState(seed)
    inds = rng_data.permutation(x_train.shape[0])
    x_train = x_train[inds]
    y_train = y_train[inds]
    x_labelled = []
    y_labelled = []
    for j in range(n_y):
        x_labelled.append(x_train[y_train == j][:num_labeled / n_y])
        y_labelled.append(y_train[y_train == j][:num_labeled / n_y])
    x_train = np.concatenate(x_labelled)
    y_train = np.concatenate(y_labelled)
    if save_dir is not None:
        y_order = np.argsort(y_train)
        x_train = x_train[y_order]
        # pile_image(x_train, 'train_data', shape=[num_labeled / n_y, n_y], path=save_dir)
    return x_train, y_train


def load_mnist_realval(path='/home/Data/mnist.pkl.gz',
                       asimage=True,
                       one_hot=False,
                       validation=True,
                       isTf=True,
                       nlabeled=None,
                       seed=None,
                       return_all=False,
                       **kwargs):
    """
    return_all flag will return all of the data. It will overwrite validation
    nlabeled.
    """
    if not os.path.isfile(path):
        data_dir = os.path.dirname(path)
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(data_dir)

        def download_dataset(url, _path):
            print('Downloading data from %s' % url)
            if sys.version_info > (2,):
                import urllib.request as request
            else:
                from urllib2 import Request as request
            request.urlretrieve(url, _path)

        download_dataset('http://www.iro.umontreal.ca/~lisa/deep/data/mnist'
                         '/mnist.pkl.gz', path)

    with gzip.open(path, 'rb') as f:
        if sys.version_info > (3,):
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        else:
            train_set, valid_set, test_set = pickle.load(f)
    x_train, y_train = train_set[0], train_set[1].astype('int32')
    x_valid, y_valid = valid_set[0], valid_set[1].astype('int32')
    x_test, y_test = test_set[0], test_set[1].astype('int32')

    n_y = y_train.max() + 1
    t_transform = (lambda x: to_one_hot(x, n_y)) if one_hot else (lambda x: x)
    y_train, y_valid = t_transform(y_train), t_transform(y_valid)
    y_test = t_transform(y_test)

    if asimage is True:
        x_train = x_train.reshape([-1, 28, 28, 1])
        x_valid = x_valid.reshape([-1, 28, 28, 1])
        x_test = x_test.reshape([-1, 28, 28, 1])
    if isTf is False:
        x_train = x_train.transpose([0, 3, 1, 2])
        x_valid = x_valid.transpose([0, 3, 1, 2])
        x_test = x_test.transpose([0, 3, 1, 2])

    if return_all is True:
        return x_train, y_train, x_valid, y_valid, x_test, y_test

    if validation is True:
        x_test = x_valid
        y_test = y_valid
    else:
        x_train = np.concatenate((x_train, x_valid))
        y_train = np.concatenate((y_train, y_valid))
    x_train, y_train = get_ssl_data(x_train, y_train, nlabeled, n_y, seed, **kwargs)
    return x_train, y_train, x_test, y_test


def load_cifar10(data_dir='/home/Data/cifar/',
                 one_hot=False,
                 isTf=True,
                 nlabelled=None,
                 seed=None,
                 **kwargs):

    def file_name(ind):
        return os.path.join(data_dir, 'cifar-10-batches-py/data_batch_' + str(ind))

    def unpickle_cifar_batch(file_):
        fo = open(file_, 'rb')
        if sys.version_info > (3,):
            tmp_data = pickle.load(fo, encoding='latin1')
        else:
            tmp_data = pickle.load(fo)
        fo.close()
        x_ = tmp_data['data'].astype(np.float32)
        x_ = x_.reshape((10000, 3, 32, 32)) / 255.
        y_ = np.array(tmp_data['labels']).astype(np.float32)
        return {'x': x_, 'y': y_}

    train_data = [unpickle_cifar_batch(file_name(i)) for i in range(1, 6)]
    x_train = np.concatenate([td['x'] for td in train_data])
    y_train = np.concatenate([td['y'] for td in train_data])
    y_train = y_train.astype('int32')

    test_data = unpickle_cifar_batch(
        os.path.join(data_dir, 'cifar-10-batches-py/test_batch'))
    x_test = test_data['x']
    y_test = test_data['y'].astype('int32')

    n_y = int(y_test.max() + 1)
    y_transform = (lambda x: to_one_hot(x, n_y)) if one_hot else (lambda x: x)

    if isTf is True:
        x_train = x_train.transpose([0, 2, 3, 1])
        x_test = x_test.transpose([0, 2, 3, 1])
    x_train, y_train = get_ssl_data(x_train, y_train, nlabelled, n_y, seed, **kwargs)
    return x_train, y_transform(y_train), x_test, y_transform(y_test)


def load_svhn(data_dir='/home/Data',
              one_hot=False,
              isTf=True,
              nlabelled=None,
              seed=None,
              **kwargs):
    data_dir = os.path.join(data_dir, 'svhn')
    train_dat = sio.loadmat(os.path.join(data_dir, 'train_32x32.mat'))
    train_x = train_dat['X'].astype('float32')
    train_y = train_dat['y'].flatten()
    train_y[train_y == 10] = 0
    train_x = train_x.transpose([3, 0, 1, 2])

    test_dat = sio.loadmat(os.path.join(data_dir, 'test_32x32.mat'))
    test_x = test_dat['X'].astype('float32')
    test_y = test_dat['y'].flatten()
    test_y[test_y == 10] = 0
    test_x = test_x.transpose([3, 0, 1, 2])

    n_y = int(train_y.max() + 1)
    y_transform = (lambda x: to_one_hot(x, n_y)) if one_hot else (lambda x: x)

    if isTf is False:
        train_x = train_x.transpose([0, 3, 1, 2])
        test_x = test_x.transpose([0, 3, 1, 2])

    train_x, train_y = get_ssl_data(train_x, train_y, nlabelled, n_y, seed, **kwargs)
    train_x, test_x = train_x / 255., test_x / 255.
    return train_x, y_transform(train_y), test_x, y_transform(test_y)

