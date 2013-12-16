"""  Model Selection Related Functions
"""
from __future__ import print_function, division, absolute_import
from .Util import log_fact_mat, warning, degree
# from SBDet import *
import numpy as np
import scipy as sp
import networkx as nx
import random
import ConfigParser
config = ConfigParser.ConfigParser()
config.read("config.ini")
# PHI_INIT_SOL = 0.1
PHI_INIT_SOL = float(config.get("models", "phi_init_sol"))
# MODEL_LIST = ["BA", "ER"]
MODEL_LIST = config.get("models", "model_list").split(",")


def sample(N, G, k):
    """  Sample degrees of k nodes in a Undirected Graph with **node** as node
    set and **edges** as edge set

    Parameters
    ---------------
    N : int
        size of graph
    G : scipy sparse matrix
        adj mat of graph
    k : int
        number of sampled nodes in graph

    Returns
    --------------
    res : array
        sampled degree values
    """
    # g = nx.Graph()
    # node_ids = range(N)
    # g.add_nodes_from(node_ids)
    # g.add_edges_from(edges)
    sel_nodes = random.sample(range(N), k)
    return degree(G)[sel_nodes]


def mg_sample(N, sigs, s_num, n_num):
    """ Multi-Graph Sample. Sample degree values from a sequence of Social
    Interaction Graphs (SIGs)

    Parameters
    ---------------
    N : int
        size of graph
    sigs : list of graphs
        each graph edge set is again a list of 2-element tuple
    s_num : int
        sampled number of sigs
    n_num : int
        sampled number of nodes in each sig

    Returns
    --------------
    s_v : 2d-matrix of ints
        each row is the sample value for a sig.

    """
    g_num = len(sigs)
    s_g = random.sample(sigs, s_num)
    s_v = np.zeros((s_num, n_num))
    for i, g in enumerate(s_g):
        s_v[i, :] = sample(N, g, n_num)

    return s_v


def _MLE_ER(deg_sample):
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
    n = np.prod(ds)
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
    # FIXME if x is int, the result is wrong
    k = np.arange(1, N)
    K, X = np.meshgrid(k, x)
    KX = np.power(K, -X)
    return -1 * np.sum(np.log(K) * KX, axis=1) / np.sum(KX, axis=1)



def _MLE_BA(deg_sample):
    # ds = deg_sample.shape
    nz_deg = deg_sample[deg_sample >= 1]
    n = len(nz_deg)
    if (n == 0):
        warning("no degree value > 1; unlikely to be BA model")
        return np.nan, -np.inf

    sl_nz_deg = np.sum(np.log(nz_deg))
    level = -1 * sl_nz_deg * 1.0 / n
    print('level', level)
    if (level >= 0):
        warning("level: %f. unlikely to be BA model" % (level))
        return np.nan, -np.inf

    th_hat = sp.optimize.newton(lambda x: phi(x + 3) - level, x0=PHI_INIT_SOL)
    if th_hat <= -1:
        warning("Estimated parameter in BA Model invalid")
        return np.nan, -np.inf
    lk = -1 * (th_hat + 3) * sl_nz_deg - n * np.log(zeta(th_hat + 3))
    return th_hat, lk

def _MLE_URN(deg_sample):
    return _MLE_BA(deg_sample+1)


def mle(deg_sample, model):
    """  Maximum Loglikelihood Estimator.

    Parameters
    ---------------
    deg_sample :
    model : str, {"BA", "ER"}
        type of estimator

    Returns
    --------------
    para : int or list
        estimated parameters
    lk : float
        log likelihood value
    """
    return globals()["_MLE_%s" % (model)](deg_sample)




def select_model(N, sigs, s_num, n_num, debug=False):
    # degrees = np.concatenate([np.array(sig.sum(axis=0)) for sig in sigs],
    # degrees = np.concatenate([degree(sig) for sig in sigs], axis=0)
    degrees = mg_sample(N, sigs, s_num, n_num)

    para_list = []
    lk_list = []
    debug_ret = dict()
    for model in MODEL_LIST:
        para, lk = mle(degrees, model)
        para_list.append(para)
        lk_list.append(lk)
        if debug:
            print('model: %s, para: %s, lk: %f' % (model, para, lk))
            debug_ret[model] = (model, para, lk)

    pos = np.argmax(lk_list)
    if debug:
        return MODEL_LIST[pos], para_list[pos], debug_ret
    return MODEL_LIST[pos], para_list[pos]


# def verify_ERGM(normal_sigs, tp, beta_values):
#     normal_dds = (degree_distribution(G, tp) for G in normal_sigs)
#     normal_dd = reduce(lambda x,y: x + y, normal_dds)
#     deg, freq = zip(*normal_dd.iteritems())
#     freq = np.array(freq, dtype=float)
#     freq /= (np.sum(freq) * 1.0)
#     deviation = lambda x: I1(freq, stats.poisson.pmf(deg, x))
#     return [deviation(b) for b in beta_values]
