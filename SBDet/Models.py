"""  Model Selection Related Functions
"""
from __future__ import print_function, division, absolute_import
from .Util import log_fact_mat, warning, degree
import numpy as np
import scipy as sp
import random
import ConfigParser
config = ConfigParser.ConfigParser()
config.read("config.ini")
PHI_INIT_SOL = float(config.get("models", "phi_init_sol"))
MODEL_LIST = config.get("models", "model_list").split(",")


def sample(N, G, k):
    """  Sample degrees of k nodes in a undirected graph with **node** as node
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
    """  Riemann zeta function
    http://en.wikipedia.org/wiki/Riemann_zeta_function
    """
    k = np.arange(1, N)
    K, X = np.meshgrid(k, x)
    KX = np.power(K, -X)
    return np.sum(KX, axis=1)


def phi(x, N=100):
    """  paritial derivative of zeta divide zeta
    """
    # FIXME if x is int, the result is wrong
    k = np.arange(1, N)
    K, X = np.meshgrid(k, x)
    KX = np.power(K, -X)
    return -1 * np.sum(np.log(K) * KX, axis=1) / np.sum(KX, axis=1)


def _MLE_PA(deg_sample, fh=lambda x: x):
    """  Maximium Likelihood Estimator of Perferential Attachment Model

    Parameters
    ---------------
    deg_sample : list or matrix
        i.i.d. samples of degrees

    Returns
    --------------
        th_hat : float
            estimated parameter
        lk : float
            log likelihood value
    """
    deg_sample = deg_sample.ravel()
    old_n = len(deg_sample)
    nz_deg = deg_sample[deg_sample >= 1]
    n = len(nz_deg)
    if (n == 0):
        warning("no degree value >= 1; unlikely to be BA model")
        return np.nan, -np.inf

    sl_nz_deg = np.sum(np.log(nz_deg))
    level = -1 * sl_nz_deg * 1.0 / n
    print('level', level)
    if (level >= 0):
        warning("level: %f. unlikely to be prefertial attachment model"
                % (level))
        return np.nan, -np.inf

    try:
        th_hat = sp.optimize.newton(lambda x: phi(fh(x)) - level,
                                    x0=PHI_INIT_SOL)
    except RuntimeError as e:
        print(e)
        return np.nan, -np.inf

    # lk = -1 * (th_hat + 3) * sl_nz_deg - n * np.log(zeta(th_hat + 3))
    # lk = -1 * (fh(th_hat)) * sl_nz_deg - n * np.log(zeta(fh(th_hat)))
    lk = -1 * (fh(th_hat)) * sl_nz_deg - old_n * np.log(zeta(fh(th_hat)))
    return th_hat, lk


def _MLE_BA(deg_sample):
    th_hat, lk = _MLE_PA(deg_sample, lambda x: x + 3.0)
    if th_hat <= -1:
        warning("Estimated parameter in BA Model invalid")
        return np.nan, -np.inf
    return th_hat, lk


def _MLE_CHJ(deg_sample):
    th_hat, lk = _MLE_PA(deg_sample, lambda x: 1.0 + 1.0 / (1.0 - x))
    if th_hat < 0 or th_hat > 1:
        warning("Estimated parameter in CHJ Model invalid")
        return np.nan, -np.inf
    return th_hat, lk


def mle(deg_sample, model):
    """  Maximum Loglikelihood Estimator.

    Parameters
    ---------------
    deg_sample :
    model : str, {"BA", "ER", "CHJ", "PA"}
        type of estimator

    Returns
    --------------
    para : int or list
        estimated parameters
    lk : float
        log likelihood value
    """
    return globals()["_MLE_%s" % (model)](deg_sample)


def select_model(degrees, model_list=None, debug=False):
    """  Select models based on samples of degrees on reference SIGs

    Parameters
    ---------------
    degrees : list of matrix
        sampled degrees

    model_list: list of str
        candidate model list. If not set, use value in config.ini

    Returns
    --------------
    model : str
        name of the selected model
    para :  float or list
        estimated parameter of the model.

    See Also
    --------------
    Use `mg_sample` to sample
    """
    model_list = model_list if model_list else MODEL_LIST

    para_list = []
    lk_list = []
    debug_ret = dict()
    for model in model_list:
        para, lk = mle(degrees, model)
        para_list.append(para)
        lk_list.append(lk)
        if debug:
            print('model: %s, para: %s, lk: %f' % (model, para, lk))
            debug_ret[model] = (model, para, lk)

    pos = np.argmax(lk_list)
    if debug:
        return model_list[pos], para_list[pos], debug_ret
    return model_list[pos], para_list[pos]
