#!/usr/bin/env python
from __future__ import print_function, division
import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from CGraph import Igraph, NetworkXGraph
# from Util import zdump,
from Util import dump, load
from Util import I1
from Util import np_index
from Util import progress_bar


def cal_SIG(data_file, interval=10.0, dur=10.0, rg=(0.0, 3000.0), directed=False):
    """ Calculate the Social Interaction Graph (SIG)

    Parameters
    ---------------
    data_file : str
        path of the flows data file
    interval, dur : float
        interval between two windows and the duration of each window
    rg : tuple (start_time, end_time)
        Only flows whose timestamps belong to [start_time, end_time) are used.
    directed : bool
        if true, the SIG is directed, otherwise it is undirected.

    Returns
    --------------
    sigs : list of NetworkX Graph

    """
    start_time, end_time = rg
    N = (end_time - start_time) // dur

    tdgs = []
    try:
        for i in xrange(N):
            progress_bar(i * 1.0 / N * 100)
            tg = NetworkXGraph(data_file)
            ips = tg.get_vertices()
            tg.add_vertices(ips)

            records = tg.filter(prot=None,
                                rg=[start_time + i * interval, start_time + i * interval + dur],
                                rg_type='time')
            edges = tg.get_edges(records)
            tg.add_edges(edges)

            tdgs.append(tg.graph)
    except:
        print('reach end')

    return sigs


def cal_cor(sigs, pivot_nodes):
    """  calculate the correlation of each node's interaction with pivot
    nodes
    """
    victim_index = np_index(ips, victim)
    for i, tg in enumerate(sigs):
        am = np.array(nx.adj_matrix(tg.graph), copy=True)
        adj_mats.append(am[:, victim_index].reshape(-1))

    npcor = np.corrcoef(adj_mats, rowvar=0)
    return npcor

    # zdump(npcor, dump_f_name)

    # np_cor_no_nan = np.nan_to_num(npcor)
    # plt.pcolor(np_cor_no_nan)
    # plt.colorbar()
    # plt.show()
    # import ipdb;ipdb.set_trace()
    pass


def animate_SIGs(sigs, ani_folder):
    layout = None
    if not os.path.exists(ani_folder):
        os.mkdir(ani_folder)

    for i, ig in enumerate(sigs):
        layout = tg.gen_layout() if layout is None else layout
        tg.plot(ani_folder + "%04d.png" % (i), layout=layout)



def visualize_caida(f_name):

    interval = 10
    dur = 10
    start_time = 0
    end_time = 30
    N = (end_time - start_time) // dur

    layout = None

    # ani_folder = './imalse_ddos_n_0_3/'
    ani_folder = './Result/merged2/'
    if not os.path.exists(ani_folder):
        os.mkdir(ani_folder)

    for i in xrange(N):
        print('i=', i)
        # tg = NetworkXGraph('./test.txt', node_info)
        tg = Igraph(f_name)
        ips = tg.get_vertices()
        tg.add_vertices(ips)
        # records = tg.filter(prot='UDP', rg=[i * interval, i * interval + dur], rg_type='time')
        records = tg.filter(prot=None,
                            rg=[start_time + i * interval, start_time + i * interval + dur],
                            rg_type='time')
        edges = tg.get_edges(records)
        tg.add_edges(edges)
        layout = tg.gen_layout() if layout is None else layout
        tg.plot(ani_folder + "%04d.png" % (i), layout=layout)

from collections import Counter


def degree_distribution(G):
    return Counter(nx.degree(G))


def entropy(D1, D2):
    K1 = D1.keys()
    K2 = D2.keys()
    K = set(K1) | set(K2)
    H1 = np.array([D1.get(key, 0) for key in K], dtype=np.float)
    # print('H1', H1)
    H1 /= (np.sum(H1) * 1.0)
    # print('H1', H1)
    H2 = np.array([D2.get(key, 0) for key in K], dtype=np.float)
    H2 /= (np.sum(H2) * 1.0)
    # print('H2', H2)
    return I1(H1, H2)


def ana_deg_distribution():
    tdg_file = './Result/merged_stored_TDG.pk'
    tdgs = load(tdg_file)
    normal_tdgs = range(200)
    dds = [degree_distribution(G) for G in tdgs]
    normal_dd = reduce(lambda x,y: x + y, (dds[idx] for idx in normal_tdgs))
    entro_list = []
    for i, dd in enumerate(dds):
        entro_list.append(entropy(dd, normal_dd))

    plt.plot(np.arange(len(entro_list)) * 10, entro_list)
    plt.xlabel('time')
    plt.ylabel('model-free entropy')
    plt.title('model-free method for degree-distribution')
    plt.savefig('./model-free-deg-dist.pdf')
    plt.show()


if __name__ == "__main__":
    # cal_corrcoef('./Result/merged_flows.csv', './Result/merged_npcor.pkz')
    # visualize_caida('./Result/merged_flows.csv')
    # cal_TDG('./Result/merged_flows.csv', './Result/merged_stored_TDG.pkz')
    # cal_TDG('./Result/merged_flows.csv', './Result/merged_stored_TDG.pk')
    ana_deg_distribution()