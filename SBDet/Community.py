#!/usr/bin/env python
""" Community Detection Related Functions
"""
from __future__ import print_function, division, absolute_import
import sys
from .Util import np, sp


def com_det(A, r_vec, w, lamb, out):
    """ Community Detection using Revised Modularity Maximization Method

    Is defined as:
        (0.5 / m) A - (0.5 / m^2) deg * deg' + w * r_rec r_rec' - lamb * I

    **deg** is the degree sequence of each node.
    **m** is the edge number of the graph

    Parameters
    ---------------
    A : symmetric matrix.
        Adjcant matrix of correlation graph
    r_vec : list
        interaction of each node with pivot nodes

    Returns
    --------------
    None
    """
    n = A.shape[0]
    deg = np.sum(A, axis=1).reshape(-1)
    mc = 0.5 * np.sum(deg)
    M = A / (2.0 * mc) - np.outer(deg, deg) / ((2.0 * mc) ** 2)
    M += w * np.outer(r_vec, r_vec) - lamb * np.eye(n)
    return max_cut(M, out)


def com_det_reg(A, r_vec, w1, w2, lamb, out):
    """ Modularity-based community detection with regularizatoin term

    The problem is defined as:
        M = (0.5 / m) A - (0.5 / m^2) deg * deg'
        max s' (M - lamb I ) s + (w1 r - w2 / 2)' s
        s.t.
            s_i^2 = 1

    **deg** is the degree sequence of each node.
    **m** is the edge number of the graph

    Parameters
    ---------------
    A : symmetric matrix.
        Adjcant matrix of correlation graph
    r_vec : list
        interaction of each node with pivot nodes

    Returns
    --------------
    P0 : M - lamb I
    q0 : w1 r - 0.5 w2 1
    W :
        | P0,     0.5 q0 |
        | 0.5 q0, 0      |
    """
    n = A.shape[0]
    deg = np.sum(A, axis=1).reshape(-1)
    mc = 0.5 * np.sum(deg)
    M = A / (2.0 * mc) - np.outer(deg, deg) / ((2.0 * mc) ** 2)
    P0 = M - lamb * np.eye(n)
    q0 = w1 * r_vec - 0.5 * w2 * np.ones((n,))
    qv = q0.reshape(-1, 1)
    zerov = np.array([[0]])
    W = np.vstack([np.hstack([P0, 0.5 * qv]),
                  np.hstack([0.5 * qv.T, zerov])])
    max_cut(W, out)
    return P0, q0, W


# def com_det_reg2(A, r_vec, w1, w2, lamb, out):
#     n = A.shape[0]
#     deg = np.sum(A, axis=1).reshape(-1)
#     mc = 0.5 * np.sum(deg)
#     M = A / (2.0 * mc) - np.outer(deg, deg) / ((2.0 * mc) ** 2)
#     P0 = M - lamb * np.eye(n) + w1 * np.outer(r_vec, r_vec)
#     q0 = - 0.5 * w2 * np.ones((n,))
#     qv = q0.reshape(-1, 1)
#     zerov = np.array([[0]])
#     W = np.vstack([np.hstack([P0, 0.5 * qv]),
#                   np.hstack([0.5 * qv.T, zerov])])
#     max_cut(W, out)
#     return P0, q0, W


def max_cut(W, out):
    """  Write the max_cut problem to **out**

    The max cut problem is:

        max Tr(XW)
        s.t.
            X >= 0
            X_{ii} = 1

    Parameters
    ---------------
    W : nxn matrix
        Weight matrix

    out : str or file handler
        specify the output file

    Returns
    --------------
    None

    """
    n = W.shape[0]
    F = [W]
    for i in xrange(n):
        dv = np.zeros((n,))
        dv[i] = 1
        F.append(np.diag(dv))
    c = np.ones((n,))
    if isinstance(out, str):
        with open(out, 'w') as fid:
            SDPA_writer(c, F, fid)
    else:
        SDPA_writer(c, F, out)


def SDPA_writer(c, F, out=sys.stdout):
    """ Write Problem to SDPA format

    See Also
    --------------
    We work with a semidefinite programming problem that has been written

    (D)    max tr(F0*Y)
           st  tr(Fi*Y)=ci           i=1,2,...,m
                     Y >= 0

    http://plato.asu.edu/ftp/sdpa_format.txt
    """
    m = len(F) - 1
    n = F[0].shape[0]
    print('%d =mdim' % (m), file=out)
    print('%d =nblocks' % (1), file=out)
    print('%d' % (n), file=out)
    print(' '.join([str(cv)for cv in c]), file=out)
    for k, f_mat in enumerate(F):
        I, J = np.triu(f_mat).nonzero()
        for i, j in zip(I, J):
            print('%d %d %d %d %f' % (k, 1, i+1, j+1, f_mat[i, j]), file=out)


def parse_CSDP_sol(f_name, n):
    """  parse CSDP solution

    Parameters
    ---------------
    f_name : str
        path of the csdp output
    n : int
        number of nodes

    Returns
    --------------
    Z :
    X : FIXME
    """
    data = np.loadtxt(f_name, skiprows=1)
    assert(np.max(data[:, 1]) == 1)
    zr, = np.where(data[:, 0] == 1)
    Z = sp.sparse.coo_matrix((data[zr, 4], (data[zr, 2]-1, data[zr, 3]-1)),
                             shape=(n, n))

    xr, = np.where(data[:, 0] == 2)
    X = sp.sparse.coo_matrix((data[xr, 4], (data[xr, 2]-1, data[xr, 3]-1)),
                             shape=(n, n))
    return Z, X


def parse_SDPA_sol(f_name, n):
    pass
    # return Y


# def randomization_old(S, W):
#     n = S.shape[0]
#     sn = 5000
#     sample = np.random.multivariate_normal(np.zeros((n,)), S.todense(),
#     (sn,))
#     val = np.zeros((sn,))
#     for i in xrange(sn):
#         fea_sol = np.sign(sample[i, :])
#         val[i] = np.dot(np.dot(fea_sol.T, W), fea_sol)
#     best_one = np.argmax(val)
#     print('the best sampled solution is :', val[best_one])
#     return np.sign(sample[best_one, :])


def randomization(S, P0, q0, sn=5000):
    """ Randomization and search a good feasible solution

    Parameters
    ---------------
    S : (n+1)x(n+1) matrix
        solution of the following relaxed problem:

        max Tr(X P0) + q0' s
        s.t
            [ X,   x]  >= 0
            [ x',  1]
            X_ii = 1

        S = [ X,   sx ]
            [ sx', 1 ]

    P0:
        M - lamb I
    q0:
        w1 r - 0.5 * w2 1


    Returns
    --------------
    sol : n-dimentional vector
        feasible solution
    """
    S = np.array(S.todense())
    X = S[:-1, :-1]
    sx = S[-1, :-1]
    covar = X - np.outer(sx, sx)
    n = X.shape[0]
    sample = np.random.multivariate_normal(np.zeros((n,)), covar, (sn,))
    val = np.zeros((sn,))
    for i in xrange(sn):
        fea_sol = np.sign(sample[i, :])
        val[i] = np.dot(np.dot(fea_sol.T, P0), fea_sol) + np.dot(q0, fea_sol)
    best_one = np.argmax(val)
    print('the best sampled solution is :', val[best_one])
    return np.sign(sample[best_one, :])


def ident_pivot_nodes(adjs, weights, thres):
    """ identify the pivot nodes

    Parameters
    ---------------
    adjs : a list of sparse matrices.
        SIG. Assume to be symmetric
    weights : a list of float
        weights of each SIG

    Returns
    --------------
    """
    N = adjs[0].shape[0]
    T = len(weights)
    total_inta_mat = np.zeros((N, T))
    for t, adj in enumerate(adjs):
        total_inta_mat[:, t] = adj.sum(axis=1).reshape(-1)
    total_inta_mat = np.dot(total_inta_mat, weights)
    total_inta_mat /= np.max(total_inta_mat)
    pivot_nodes, = np.where(total_inta_mat > thres)
    return pivot_nodes


def cal_inta_pnodes(adjs, weights, pivot_nodes):
    """  calculate the interactions of nodes with pivot_nodes using GCM

    Parameters
    ---------------
    adjs : a list of sparse matrices.
        SIG. Assume to be symmetric
    weights : a list of float
        weights of each SIG
    pivot_nodes : list of ints
        a set of nodes that may be leaders or the victims of the botnet

    Returns
    --------------
    """
    N = adjs[0].shape[0]
    T = len(weights)
    inta_mat = np.zeros((N, T))
    for t, adj in enumerate(adjs):
        res = np.sum(adj[list(pivot_nodes), :].todense(), axis=0).reshape(-1)
        inta_mat[:, t] = res
    inta = np.dot(inta_mat, weights)
    return inta


# def cal_cor(sigs, pivot_nodes):
#     """  calculate the correlation of each node's interaction with pivot
#     nodes
#     """
#     ips = sigs[0].get_vertices()
#     victim_index = np_index(ips, pivot_nodes)
#     adj_mats = []
#     for i, tg in enumerate(sigs):
#         am = np.array(nx.adj_matrix(tg.graph), copy=True)
#         adj_mats.append(am[:, victim_index].reshape(-1))

#     npcor = np.corrcoef(adj_mats, rowvar=0)
#     return npcor

def cal_cor_graph(adjs, pivot_nodes, thres):
    """  calculate the correlation graph

    Parameters
    ---------------
    adjs : a list of sparse matrices.
        SIG. Assume to be symmetric
    pivot_nodes : list of ints
        a set of nodes that may be leaders or the victims of the botnet
    thres : float
        threshold for constructing correlation graph

    Returns
    --------------
    A : np.2darray
        adj matrix of the correlation graph
    npcor : np.2darray
        matrix of correlation coefficients.
    """
    inta = lambda x: np.sum(np.array(adj[pivot_nodes, :].todense()),
                            axis=0).reshape(-1)
    traf = np.asarray([inta(adj) for adj in adjs])
    npcor = np.corrcoef(traf, rowvar=0)
    np_cor_no_nan = np.nan_to_num(npcor)
    A = np_cor_no_nan > thres
    return A, npcor
