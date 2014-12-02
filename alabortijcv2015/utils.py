from __future__ import division
import cPickle
from skimage.filter import gaussian_filter


def pickle_load(path):
    with open(str(path), 'rb') as f:
        return cPickle.load(f)


def pickle_dump(obj, path):
    with open(str(path), 'wb') as f:
        cPickle.dump(obj, f, protocol=2)


def flatten_out(list_of_lists):
    return [i for l in list_of_lists for i in l]


fsmooth = lambda x, sigma: gaussian_filter(x, sigma, mode='constant')
