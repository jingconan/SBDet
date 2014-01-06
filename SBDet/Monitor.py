"""  Functions related to network anomaly detection.
"""
from __future__ import print_function, division, absolute_import
from .Util import adjust_pv, degree, KL_div, xlogx
import numpy as np
from scipy.stats import poisson


def divergence(dd, gtp, para):
    """  Calculate the divergenc a degree distribution with respect to model
    **gtp** with parameter **para**

    Parameters
    ---------------
    dd : vector
        degree distribution
    gtp : str, {'ER', 'BA'}
        graph type
    para : list
        parameters of the model

    Returns
    --------------
    res : float
        divergence value
    """
    if not isinstance(para, list) and \
            not isinstance(para, tuple):
        para = [para]
    return globals()['_I_' + gtp](dd, *para)


def get_deg_dist(G, minlength=None):
    """  Get degree distribution from a graph

    Parameters
    ---------------
    G : sparse matrix
        adj matrix of an undirected graph. up-trigular
    minlength : int
        mini length of the result

    Returns
    --------------
    ret : 1-d array
        len(ret) >= minlength
    """
    hist = np.bincount(degree(G), minlength=minlength)
    ret = np.array(hist, dtype=float) / np.sum(hist)
    assert(np.min(ret) >= 0 and np.max(ret) <= 1)
    return ret


def monitor_deg_dis(sigs, gtp, para, minlength=None):
    """  Monitor the degree distribution

    Parameters
    ---------------
    sigs : list of sparse matrix
        each sparse matrix is adj matrix of a **undirected** Social
        Interaction graph. Assume to be up-trigular
    gtp : str {'ER', 'BA', 'PA', 'CHJ'}
        graph type
    para : list
        list of parameters
    minlength : int, optional
        A minimum number of bins for the degree distribution

    Returns
    --------------
    divs : list
        list of divergence value for each sig.
    """
    return [divergence(get_deg_dist(G, minlength), gtp, para) for G in sigs]


def _aux_I_PA(dd, p, beta):
    if len(dd) == 1:  # no interaction
        return 0
    dd = dd[dd > 0]
    d = len(dd)
    assert(np.max(dd) <= 1)
    assert(np.min(dd) > 0)
    cgamma = np.cumsum(dd)
    crit = np.sum(1 - cgamma)
    assert(abs(np.sum(dd) - 1.0) < 1e-3)
    if crit > 1:
        print("[warning]: invalid degree distribution, "
              "you may need to descrese eps")
        return -np.inf  # unlikely to be URN Model

    s0 = xlogx(1 - dd[0]) + \
        (1 - dd[0]) * (- np.log(p + (1-p) * beta * dd[0] / (1.0 + beta)))
    C = - np.log(1.0 - p) - np.log(np.arange(1, d) + beta) + \
        np.log(1 + beta) - np.log(dd[1])
    s1 = xlogx(1-cgamma[1:]) + np.sum((1 - cgamma[1:]) * C)
    s2 = (1 - crit) * (np.log(1.0 + beta) - np.log(1.0 - p))
    return s0 + s1 + s2


def _I_BA(dd, alpha, eps):
    """  Rate Function for BA Model

    Parameters
    ---------------
    dd : vector
        observed distribution distribution
    alpha : float
        parameter of BA model. Every new node is attached existin
        node. The probability of an existing node i be attached is proportional
        to d_i + alpha, where d_i is the degree of node i.
    eps : float
        we will adjust the degre distribution so that
            min(dd) >= eps

    Returns
    --------------
    res : float
        rate value of distribution *dd* for BA model with parameter *alpha*

        Rate function is defined as
        I = sum((1 - [dd]_i) log ((1-[dd]_i) / (i+1) (dd_i / 2))) + \
                (1 - sum(i dd_i)) log2

    See Also
    -----------
        Choi, J., & Sethuraman, S. (2011). Large deviations in preferential
        attachment schemes.Choi, J., & Sethuraman, S. (2011). Large deviations
        in preferential attachment schemes.
    """
    dd = adjust_pv(dd, eps)  # min(dd) >= eps
    return _aux_I_PA(dd, 0, alpha + 1)


def _I_CHJ(dd, p, eps):
    dd = adjust_pv(dd, eps)  # min(dd) >= eps
    assert(p < 1 and p > 0)
    return _aux_I_PA(dd, p, 1)


def _I_PA(dd, alpha, p, eps):
    return min(_I_BA(dd, alpha, eps), _I_CHJ(dd, p, eps))


# def _I_URN(dd, beta, eps):
#     dd = dd[1:]
#     dd /= np.sum(dd)
#     dd = adjust_pv(dd, eps)  # min(dd) >= eps
#     return _I_BA(dd, beta, eps)


def _I_ER(dd, beta, eps):
    n = len(dd)
    mu_bar = np.dot(np.arange(n), dd)
    poisson_pdf = poisson.pmf(range(n), beta)
    return KL_div(dd, poisson_pdf, eps) + 0.5 * (mu_bar - beta) + \
        0.5 * mu_bar * np.log(beta) - \
        0.5 * xlogx(mu_bar)
