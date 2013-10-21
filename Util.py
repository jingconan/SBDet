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


def load(f_name):
    with open(f_name, 'rb') as f:
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


import matplotlib.pyplot as plt
import networkx as nx


def save_graph(graph,file_name):
    #initialze Figure
    plt.figure(num=None, figsize=(20, 20), dpi=80)
    plt.axis('off')
    fig = plt.figure(1)
    pos = nx.spring_layout(graph)
    nx.draw_networkx_nodes(graph,pos)
    nx.draw_networkx_edges(graph,pos)
    nx.draw_networkx_labels(graph,pos)

    cut = 1.00
    xmax = cut * max(xx for xx, yy in pos.values())
    ymax = cut * max(yy for xx, yy in pos.values())
    plt.xlim(0, xmax)
    plt.ylim(0, ymax)

    plt.savefig(file_name,bbox_inches="tight")
    plt.close()
    del fig

import time

START_TIME = -1

def log(*args):
    if globals()['START_TIME']== -1:
        globals()['START_TIME']= time.time()
    msg = ' '.join([str(a) for a in args])
    print('[%f s]: --> %s' % (time.time() - globals()['START_TIME'], msg))




def adjust_pv(prob, eps):
    """ adjust probability vector so that each value >= eps

    Parameters
    ---------------
    prob : list or tuple
        probability vector
    eps : float
        threshold

    Returns
    --------------
    prob : list
        adjusted probability vector

    Examples
    -------------------
    >>> adjust_pv([0, 0, 1], 0.01)
    [0.01, 0.01, 0.98]

    """
    assert(abs(sum(prob) - 1) < 1e-3)
    a = len(prob)
    zei = [i for i, v in zip(xrange(a), prob) if abs(v) < eps] # zero element indices
    if a == len(zei): # all elements are zero
        return [eps] * a
    zei_sum = sum(prob[i] for i in zei)
    adjustment = (eps * len(zei) - zei_sum) * 1.0 / (a - len(zei))
    prob2 = [v - adjustment for v in prob]
    for idx in zei:
        prob2[idx] = eps
    if min(prob2) < 0:
        print( '[warning] EPS is too large in adjust_pv')
        import pdb;pdb.set_trace()
        # return adjust_pv(prob2, eps / 2.0)
    return prob2




EPS = 1e-20
from math import log
def I1(nu, mu):
    """  Calculate the empirical measure of two probability vector nu and mu

    Parameters
    ---------------
    nu, mu : list or tuple
        two probability vector

    Returns
    --------------
    res : float
        the cross entropy

    Notes
    -------------
    The cross-entropy of probability vector **nu** with respect to **mu** is
    defined as

    .. math::

        H(nu|mu) = \sum_i  nu(i) \log(nu(i)/mu(i)))

    One problem that needs to be addressed is that mu may contains 0 element.

    Examples
    --------------
    >>> print I1([0.3, 0.7, 0, 0], [0, 0, 0.3, 0.7])
    45.4408375578

    """
    assert(len(nu) == len(mu))
    a = len(nu)

    mu = adjust_pv(mu, EPS)
    nu = adjust_pv(nu, EPS)

    H = lambda x, y:x * log( x * 1.0 / y )
    return sum(H(a, b) for a, b in zip(nu, mu))


