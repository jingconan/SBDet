from __future__ import print_function, division, absolute_import
# from .Util import I1, adjust_pv
from .Util import adjust_pv
import numpy as np
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


def _I_BA(dd, alpha):
    """  Rate Function for BA Model

    Parameters
    ---------------
    dd : vector
        observed distribution distribution
    alpha : float
        parameter of BA model. Every new node is attached existin
        node. The probability of an existing node i be attached is proportional
        to d_i + alpha, where d_i is the degree of node i.

    Returns
    --------------
    res : float
        rate value of distribution *dd* for BA model with parameter *alpha*
    """
    d = len(dd)
    cgamma = np.cumsum(dd)
    crit = np.sum(1 - cgamma)
    assert(abs(np.sum(dd) - 1.0) < 1e-3)
    assert(crit <= 1)

    C = (1 - cgamma) / (np.arange(d) + 1 + alpha)
    C /= (dd / (2 + alpha))
    s1 = np.nansum((1 - cgamma) * np.log(C))
    s2 = (1 - crit) * np.log(2 + alpha)
    return s1 + s2


def _I_ER(dd, beta):
    # Stub
    pass


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
    return locals()['_I_' + gtp](dd, *para)


def get_deg_dist(G, minlength=100, eps=1e-06):
    """  Get degree distribution from a graph

    Parameters
    ---------------
    G : sparse matrix
        adj matrix of an undirected graph. up-trigular
    minlength : int
        mini length of the result
    eps : float > 0
        mini value in the result

    Returns
    --------------
    dd : 1-d array
        len(dd) >= minlength
        min(dd) >= eps
    """
    hist = np.bincount(G.sum(axis=0), minlength=minlength)[1:]
    hist = np.array(hist, dtype=float) / np.sum(hist)
    return adjust_pv(hist, eps)


def monitor_deg_dis(sigs, gtp, para):
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

    Returns
    --------------
    divs : list
        list of divergence value for each sig.
    """
    return [divergence(get_deg_dist(G), gtp, para) for G in sigs]
