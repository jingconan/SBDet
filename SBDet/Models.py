"""  Model Selection Related Functions
"""
from __future__ import print_function, division, absolute_import
from .Util import log_fact_mat, warning
# from SBDet import *
import numpy as np
import scipy as sp
import networkx as nx
import random

def sample(nodes, edges, k):
    """  Sample degrees of k nodes in a Undirected Graph with **node** as node set and
    **edges** as edge set

    Parameters
    ---------------
    nodes : list
        set of nodes
    edge : list of tuple with two-element
        set of edges

    Returns
    --------------
    res : list of ints
        sampled degree values
    """
    g = nx.Graph()
    node_ids = range(len(nodes))
    g.add_nodes_from(node_ids)
    g.add_edges_from(edges)
    return list(g.degree(random.sample(node_ids, k)).values())

def mg_sample(nodes, sig_edges, n, k):
    """ Multi-Graph Sample. Sample degree values from a sequence of Social Interaction Graphs
    (SIGs)

    Parameters
    ---------------
    nodes : list
        set of nodes
    sig_edges : list of graph edge set
        each graph edge set is again a list of 2-element tuple
    n : int
        sampled number of sigs
    k : int
        sampled number of nodes in each sig

    Returns
    --------------
    s_v : 2d-matrix of ints
        each row is the sample value for a sig.

    """
    g_num = len(sig_edges)
    s_g = random.sample(range(g_num), n)
    s_v = np.zeros((n, k))
    for i, g in enumerate(s_g):
        edges = zip(*sig_edges[g])
        if not edges:
            continue
        edges = edges[0]
        s_v[i, :] = sample(nodes, edges, k)

    return s_v

def _ER_MLE(deg_sample):
    """  Maximium log likelihood estimator for ER model

    Parameters
    ---------------
    deg_sample : 1d or 2d array of ints
        samples of degrees

    Returns
    --------------
    th_hat : float
        estimated parameter for ER model
    lk : float
        loglikelihood value.
    """
    ds = deg_sample.shape
    n = ds[0] * ds[1]
    th_hat = np.sum(deg_sample) * 1.0 / n
    # pm = th_hat * np.ones(ds)
    lk = np.log(th_hat) * np.sum(deg_sample) - th_hat * n - \
            np.sum(log_fact_mat(deg_sample))
    return th_hat, lk


def zeta(x, N=100):
    k = np.arange(1, N)
    K, X = np.meshgrid(k, x)
    KX = np.power(K, -X)
    return np.sum(KX, axis=1)

def phi(x, N=100):
    k = np.arange(1, N)
    K, X = np.meshgrid(k, x)
    KX = np.power(K, -X)
    return -1 * np.sum(np.log(K) * KX, axis=1) / np.sum(KX, axis=1)

def _BA_MLE(deg_sample):
    # ds = deg_sample.shape
    nz_deg = deg_sample[deg_sample >= 1]
    n = len(nz_deg)
    if (np.max(nz_deg) <= 1):
        warning("no degree value > 1; unlikely to be BA model")
        return np.nan, -np.inf

    sl_nz_deg = np.sum(np.log(nz_deg))
    level = -1 * sl_nz_deg * 1.0 / n
    print('level', level)
    th_hat = sp.optimize.newton(lambda x: phi(x + 3) - level , x0=3)
    lk = -1 * (th_hat + 3) * sl_nz_deg - n * np.log(zeta(th_hat + 3))
    return th_hat, lk

def mle(deg_sample, model):
    model_dict = {
            "BA": _BA_MLE,
            "ER": _ER_MLE,
            }
    return model_dict[model](deg_sample)

