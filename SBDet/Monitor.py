from __future__ import print_function, division, absolute_import
# from .Util import I1, adjust_pv
from .Util import adjust_pv, degree, KL_div, xlogx
import numpy as np
from scipy.stats import poisson
# from collections import Counter
# import networkx as nx


# def _vector_represent_dd(D1, D2):
#     """  Represent Counters as vector

#     Parameters
#     ---------------
#     D1, D2 : counter class

#     Returns
#     --------------
#     H1, H2 : 1-d vector
#         the length of the vector == size of union set of keys in D1 and D2
#     """
#     K1 = D1.keys()
#     K2 = D2.keys()
#     K = set(K1) | set(K2)
#     H1 = np.array([D1.get(key, 0) for key in K], dtype=np.float)
#     H1 /= (np.sum(H1) * 1.0)
#     H2 = np.array([D2.get(key, 0) for key in K], dtype=np.float)
#     H2 /= (np.sum(H2) * 1.0)
#     return H1, H2


# def KL_div(D1, D2):
#     """  KL divergence of two degree distribution

#     Parameters
#     ---------------
#     D1, D2 : Counter
#         frequency of each degree

#     Returns
#     --------------
#     res : float
#         the Kullback-Leibler divergence between D1 and D2
#     """
#     H1, H2 = _vector_represent_dd(D1, D2)
#     return I1(H1, H2)


def cal_mean_deg(dd):
    total_freq = 0
    sum_deg = 0
    for d, f in dd.iteritems():
        sum_deg += f * d
        total_freq += f
    sum_deg /= (total_freq * 1.0)
    return sum_deg


# def LDP_div(D1, D2):
#     H1, H2 = _vector_represent_dd(D1, D2)
#     m1 = cal_mean_deg(D1)
#     m2 = cal_mean_deg(D2)
#     return I1(H1, H2) + 0.5 * (m1 - m2) + \
#             0.5 * m1 * np.log(m2) - \
#             0.5 * m1 * np.log(m1)


# def degree_distribution(G, tp):
#     if tp == "igraph":
#         return Counter(G.degree())
#     elif tp == 'networkx':
#         return Counter(nx.degree(G))
#     else:
#         raise Exception("Unknown type of graph")


def _GenURN(dd, p, beta):
    if len(dd) ==1: # no interaction
        return 0
    dd = dd[dd>0]
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
            (1 - dd[0]) * (- np.log(p + (1-p) * beta * dd[0] / (1.0 + beta)) )
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
    return _GenURN(dd, 0, alpha + 1)

    #ignore the zero degrees
    # if alpha < -1:
    #     raise Exception("alpha=%f. alpha in BA Model should > -1" % (alpha))
    # dd = dd[dd > 0]
    # d = len(dd)
    # cgamma = np.cumsum(dd)
    # crit = np.sum(1 - cgamma)
    # assert(abs(np.sum(dd) - 1.0) < 1e-3)
    # if crit > 1:
    #     print("[warning]: invalid degree distribution, "
    #           "you may need to descrese eps")
    #     return -np.inf  # unlikely to be BA Model

    # C = (1 - cgamma) / (np.arange(d) + 1 + alpha)
    # C /= (dd / (2 + alpha))
    # s1 = np.nansum((1 - cgamma) * np.log(C))
    # s2 = (1 - crit) * np.log(2 + alpha)
    # return s1 + s2

def _I_CHJ(dd, p, eps):
    dd = adjust_pv(dd, eps)  # min(dd) >= eps
    assert(p < 1 and p > 0)
    return _GenURN(dd, p, 1)


def _I_URN(dd, beta, eps):
    dd = dd[1:]
    dd /= np.sum(dd)
    dd = adjust_pv(dd, eps)  # min(dd) >= eps
    return _I_BA(dd, beta, eps)


def _I_ER(dd, beta, eps):
    n = len(dd)
    mu_bar = np.dot(np.arange(n), dd)
    poisson_pdf = poisson.pmf(range(n), beta)
    return KL_div(dd, poisson_pdf, eps) + 0.5 * (mu_bar - beta) + \
        0.5 * mu_bar * np.log(beta) - \
        0.5 * mu_bar * np.log(mu_bar)



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
    dd : 1-d array
        len(dd) >= minlength
        min(dd) >= eps
    """
    # import ipdb;ipdb.set_trace()
    # hist = np.bincount(degree(G), minlength=minlength)[1:]
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
    gtp : str {'ER', 'BA'}
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
