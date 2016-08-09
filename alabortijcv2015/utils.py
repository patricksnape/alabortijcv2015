from __future__ import division
try:
    import cPickle as pickle
except ImportError:
    import pickle
from skimage.filters import gaussian_filter


def pickle_load(path):
    with open(str(path), 'rb') as f:
        return pickle.load(f)


def pickle_dump(obj, path):
    with open(str(path), 'wb') as f:
        pickle.dump(obj, f, protocol=2)


def flatten_out(list_of_lists):
    return [i for l in list_of_lists for i in l]


fsmooth = lambda x, sigma: gaussian_filter(x, sigma, mode='constant')
