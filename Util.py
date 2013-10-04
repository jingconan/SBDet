def abstract_method():
    """ This should be called when an abstract method is called that should have been
    implemented by a subclass. It should not be called in situations where no implementation
    (i.e. a 'pass' behavior) is acceptable. """
    raise NotImplementedError('Method not implemented!')


def binary_search(a, x, lo=0, hi=None):
    """
    Find the index of largest value in a that is smaller than x.
    a is sorted Binary Search
    """
    # import pdb;pdb.set_trace()
    if hi is None:
        hi = len(a)
    while lo < hi:
        mid = (lo + hi) // 2
        midval = a[mid]
        if midval < x:
            lo = mid + 1
        elif midval > x:
            hi = mid
        else:
            return mid
    return hi - 1

Find = binary_search


class DataEndException(Exception):
    pass

import numpy as np


try:
    import cPickle as pickle
except ImportError:
    import pickle


def dump(obj, f_name):
    with open(f_name, 'wb') as f:
        pickle.dump(obj,f)


def load(obs, f_name):
    with open(f_name, 'wb') as f:
        return pickle.load(f)

import gzip
proto = pickle.HIGHEST_PROTOCOL


def zdump(obj, f_name):
    f = gzip.open(f_name,'wb', proto)
    pickle.dump(obj,f)
    f.close()


def zload(f_name):
    f = gzip.open(f_name,'rb', proto)
    obj = pickle.load(f)
    f.close()
    return obj


