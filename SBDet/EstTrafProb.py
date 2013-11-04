#!/home/wangjing/Apps/python/bin/python
#!/usr/bin/env python

"""  Estimate the Null Model using Entropy Maximization

Parameters
---------------
Returns
--------------
"""
#############
from __future__ import print_function, division, absolute_import
import numpy as np
import openopt as opt

from .Util import progress_bar
from .Util import load, dump
# import networkx as nx
import matplotlib.pyplot as plt


def QP(yp, c_vec, N, T, solver='cplex', H_vec=None, b_vec=None):
    # the shape of c_vec is  (N*N, T)
    a_vec = np.dot(c_vec, yp)

    hc = 1.0 / a_vec # scale coef for H
    hc[hc == np.inf] = 0

    hct = np.tile(np.sqrt(hc).reshape(-1, 1), (1, T))
    c_vec_scale = c_vec * hct
    H = np.einsum('ij,ik->jk', c_vec_scale, c_vec_scale)

    bc = np.log(a_vec) - 1
    bc[bc == -np.inf] = 0
    b = np.dot(bc, c_vec)

    print('finish calculating H, b')

    p = opt.QP(H=1 * H,
               f=1 * b,
               Aeq=np.ones((T,)),
               beq=1.0,
               lb=0.0 * np.zeros((T,)),
               ub=1.0 * np.ones((T,)))
    r = p._solve(solver, iprint=0)
    f_opt, x_opt = r.ff, r.xf
    print('x_opt', x_opt)
    print('f_opt', f_opt)
    return f_opt, x_opt

def test():
    yp = np.array([1.0, 0.0])
    c_vec = np.zeros((2.0, 2.0, 2.0))
    c_vec[0, 0, :] = [3.0, 1.0]
    c_vec[0, 1, :] = [2.0, 1.0]
    c_vec[1, 0, :] = [3.0, 4.0]
    c_vec[1, 1, :] = [3.0, 2.0]

    QP(yp, c_vec, 'cvxopt_qp')

#############
import scipy.sparse as sps


def EstTrafProb(adj_mats, T=None, eps=1e-5):
    if T is not None:
        adj_mats = adj_mats[:T]

    N = adj_mats[0].shape[0]  # size of the graph
    T = len(adj_mats)   # number of graphs
    # calculate C from degrees
    # k_in and k_out are NxT array
    k_in = np.vstack([mat.sum(axis=0) for mat in adj_mats]).T
    k_out = np.hstack([mat.sum(axis=1) for mat in adj_mats])
    m = np.array(k_in.sum(axis=0)).reshape(-1)  # m is 1xT array
    m2 = (m ** 2)
    c_vec = np.zeros((N * N, T))
    for t in xrange(T):
        mat = np.outer(k_out[:, t], k_in[:, t]) * 1.0 / m2[t]
        c_vec[:, t] = mat.reshape(-1)
    print('finish calculate c_vec')
    # yp = 1.0 / T * np.ones((T,))
    yp = np.random.rand(T)
    yp /= np.sum(yp)
    it_n = -1
    tr = dict()
    tr['err'] = []
    while it_n < 10:
        it_n += 1
        print('iteration: ', it_n)

        f_obj, y_new = QP(yp, c_vec, N, T, 'cplex')
        err = np.sum(np.abs(y_new - yp))
        print('err', err)
        tr['err'].append(err)
        if err < eps:
            break
        yp = y_new
    tr['solution'] = yp
    tr['f_obj'] = f_obj
    return tr
    # plt.plot(tr['err'])
    # plt.show()
    # import ipdb;ipdb.set_trace()

import copy
def EstTrafProbBatch(adj_mats, T=10):
    TT =  len(adj_mats)
    print('TT', TT)
    st = 0
    tr = dict()
    tr['batch'] = []
    sol = []
    for i in xrange(0, TT//T):
        print('*'*20)
        print('i: ', i)
        print('*'*20)
        new_adj = copy.deepcopy(adj_mats[(i*T):((i+1)*T)])
        tr_b = EstTrafProb(new_adj)
        tr['batch'].append(tr_b)
        sol.append(tr_b['solution'])

    import ipdb;ipdb.set_trace()
    tr['solution'] = np.concatenate(sol)





# def main():
    # adj = load('./Result/merged_stored_TDG.pkz')
    # pass
# main()
# graphs = load('./Result/merged_stored_TDG.pk')

# adj_mats = [nx.to_scipy_sparse_matrix(g) for g in graphs]
# dump(adj_mats, './Result/merged_TDG_sparse_adjs.pk')

# calculate probability to have traffic from node i to j.
# adj_mats = load('./Result/merged_TDG_sparse_adjs.pk')
# adj_mats = adj_mats[0:5]
# dump(adj_mats, './Result/merged_TDG_sparse_adjs_first_5.pk')

import time
def cal_time_complexity_with_T():
    # adj_mats = load('./Result/merged_TDG_sparse_adjs_first_5.pk')
    adj_mats = load('./Result/merged_TDG_sparse_adjs.pk')
    tr = dict()
    tr['T'] = range(5, 50, 5)
    tr['run_time'] = []
    tr['run_tr'] = []
    for t in tr['T']:
        print('**'*20)
        print('t, ', t)
        print('**'*20)
        st = time.time()
        tr_run = EstTrafProb(adj_mats, t)
        et = time.time()
        tr['run_time'].append(et - st)
        tr['run_tr'].append(tr_run)
    dump(tr, './Result/time_complexity_with_T.pk')

# cal_time_complexity_with_T()


plt.rc('text', usetex=True)
plt.rc('font', family='serif')
def plot_EM_err(f_name):
    # tr = load('./Result/EM_simple_trace.pk')
    tr = load(f_name)
    plt.plot(tr['err'], '+-')
    plt.title('$|y_{k+1} - y_k|$')
    plt.xlabel('iteration')
    plt.ylabel('difference')
    plt.savefig('EM_err.pdf')
    plt.show()


# plot_EM_err('./Result/EM_simple_trace.pk')
########################

def plot_time_complexity():
    tr = load('./Result/time_complexity_with_T.pk')
    plt.plot(tr['T'], tr['run_time'], 'b-+')
    plt.title('time vs no. of observed TDGs')
    plt.xlabel('no. of observed TDGs')
    plt.ylabel('time used by EM')
    plt.savefig('time_vs_num_TDG.pdf')
    plt.show()

# plot_time_complexity()
########################
