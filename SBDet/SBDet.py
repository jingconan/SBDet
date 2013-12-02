#!/usr/bin/env python
from __future__ import print_function, division, absolute_import
import os
import sys
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter


from .Data import HDF_FlowExporter
# from Dataset import HDF_FlowExporter
from .CGraph import Igraph
from .CGraph import NetworkXGraph
from .Util import dump, load
from .Util import I1
from .Util import np_index, np_to_dotted
from .Util import progress_bar
from .Util import DataEndException
from .Util import igraph
from .Util import sp, la, stats


def cal_SIG_low_mem(data_file, interval=10.0, dur=10.0,
                    rg=(0.0, float('inf')), folder=None):
    """ Calculate the Social Interaction Graph (SIG)

    Parameters
    ---------------
    data_file :
        flows data file
    interval, dur : float
        interval between two windows and the duration of each window
    rg : tuple (start_time, end_time)
        Only flows whose timestamps belong to [start_time, end_time) are used.
    folder : str
        path for the output folder
    """
    start_time, end_time = rg
    N = sys.maxint if end_time == float('inf') \
        else int((end_time - start_time) // dur)

    ips = NetworkXGraph(data_file).get_vertices()
    dump(ips, folder + 'nodes.pk')
    try:
        for i in xrange(N):
            sys.stdout.write("[%d]" % i)
            sys.stdout.flush()

            tg = NetworkXGraph(data_file)
            tg.add_vertices(ips)
            seg = [start_time + i * interval, start_time + i * interval + dur]
            records = tg.filter(prot=None, rg=seg, rg_type='time')
            edges = tg.get_edges(records)
            tg.add_edges(edges)
            dump({'edges': edges}, '%s%i.pk' % (folder, i))
    except DataEndException:
        print('reach end')

    sys.stdout.write('\n')



def pack_sigs(folder, out):
    """  pack the sequence of sigs in a **folder** into a single file
    """
    N = sys.maxint
    try:
        nodes = load('%snodes.pk' % (folder))
        nm = dict(zip(nodes, range(len(nodes))))
        sigs = []
        for i in xrange(N):
            if i % 100 == 0:
                sys.stdout.write("\r%d" % i)
                sys.stdout.flush()
            dat = load('%s%i.pk' % (folder, i))
            ce = lambda edge: (nm[edge[0]], nm[edge[1]])
            sigs.append([(ce(e), c) for e, c in dat['edges'].iteritems()])
    except IOError:
        pass
    dump({'nodes': nodes, 'sig_edges': sigs}, out)
    sys.stdout.write('\n')



def cal_SIG(data_file, interval=10.0, dur=10.0, rg=(0.0, float('inf')),
            directed=False, tp='igraph'):
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
    tp : {'igraph', 'networkx'}
        type of graph

    Returns
    --------------
    sigs : list of Graphs with format specified by tp

    """
    start_time, end_time = rg
    if end_time == float('inf'):
        N = sys.maxint
    else:
        N = int((end_time - start_time) // dur)

    sigs = []
    if isinstance(data_file, str):
        data = HDF_FlowExporter(data_file)
    else:
        data = data_file

    # TGraph = NetworkXGraph
    TGraph_map = {
        'igraph': Igraph,
        'networkx': NetworkXGraph,
    }
    TGraph = TGraph_map[tp]

    ips = TGraph(data).get_vertices()
    # ips = [np_to_dotted(ip) for ip in ips]
    try:
        for i in xrange(N):
            progress_bar(i * 1.0 / N * 100)
            tg = TGraph(data)
            tg.add_vertices(ips)
            records = tg.filter(prot=None,
                                rg=[start_time + i * interval, start_time + i * interval + dur],
                                rg_type='time')
            edges = tg.get_edges(records)
            # tg.add_edges(edges)
            sigs.append(tg.graph)
    except DataEndException:
        print('reach end')

    return sigs


def cal_cor(sigs, pivot_nodes):
    """  calculate the correlation of each node's interaction with pivot
    nodes
    """
    ips = sigs[0].get_vertices()
    victim_index = np_index(ips, pivot_nodes)
    adj_mats = []
    for i, tg in enumerate(sigs):
        am = np.array(nx.adj_matrix(tg.graph), copy=True)
        adj_mats.append(am[:, victim_index].reshape(-1))

    npcor = np.corrcoef(adj_mats, rowvar=0)
    return npcor


def animate_SIGs(sigs, ani_folder):
    if not isinstance(sigs[0], igraph.Graph):
        raise Exception("animate_SIGs only works with python-igraph")
    layout = None
    if not os.path.exists(ani_folder):
        os.mkdir(ani_folder)

    N = len(sigs)
    print('animation progress:')
    for i, ig in enumerate(sigs):
        progress_bar(i * 1.0 / N * 100)
        # nig = NetworkXGraph(graph=ig)
        nig = Igraph(graph=ig)
        layout = nig.gen_layout() if layout is None else layout
        nig.plot(ani_folder + "%04d.png" % (i), layout=layout)


def degree_distribution(G, tp):
    if tp == "igraph":
        return Counter(G.degree())
    elif tp == 'networkx':
        return Counter(nx.degree(G))
    else:
        raise Exception("Unknown type of graph")


def vector_represent_dd(D1, D2):
    K1 = D1.keys()
    K2 = D2.keys()
    K = set(K1) | set(K2)
    H1 = np.array([D1.get(key, 0) for key in K], dtype=np.float)
    H1 /= (np.sum(H1) * 1.0)
    H2 = np.array([D2.get(key, 0) for key in K], dtype=np.float)
    H2 /= (np.sum(H2) * 1.0)
    return H1, H2

def KL_div(D1, D2):
    """  KL divergence of two degree distribution

    Parameters
    ---------------
    D1, D2 : Counter
        frequency of each degree

    Returns
    --------------
    res : float
        the Kullback-Leibler divergence between D1 and D2
    """
    H1, H2 = vector_represent_dd(D1, D2)
    return I1(H1, H2)

def cal_mean_deg(dd):
    total_freq = 0
    sum_deg = 0
    for d, f in dd.iteritems():
        sum_deg += f * d
        total_freq += f
    sum_deg /= (total_freq * 1.0)
    return sum_deg


def LDP_div(D1, D2):
    H1, H2 = vector_represent_dd(D1, D2)
    m1 = cal_mean_deg(D1)
    m2 = cal_mean_deg(D2)
    return I1(H1, H2) + 0.5 * (m1 - m2) + \
            0.5 * m1 * np.log(m2) - \
            0.5 * m1 * np.log(m1)


def monitor_deg_dis(sigs, normal_sigs, tp, div_func):
    dds = (degree_distribution(G, tp) for G in sigs)
    normal_dds = (degree_distribution(G, tp) for G in normal_sigs)
    normal_dd = reduce(lambda x,y: x + y, normal_dds)
    divs = [div_func(dd, normal_dd) for dd in dds]
    return divs

    # plt.plot(np.arange(len(entro_list)) * 10, entro_list)
    # plt.xlabel('time')
    # plt.ylabel('model-free entropy')
    # plt.title('model-free method for degree-distribution')
    # plt.savefig('./model-free-deg-dist.pdf')
    # plt.show()

def verify_ERGM(normal_sigs, tp, beta_values):
    normal_dds = (degree_distribution(G, tp) for G in normal_sigs)
    normal_dd = reduce(lambda x,y: x + y, normal_dds)
    deg, freq = zip(*normal_dd.iteritems())
    freq = np.array(freq, dtype=float)
    freq /= (np.sum(freq) * 1.0)
    deviation = lambda x: I1(freq, stats.poisson.pmf(deg, x))
    return [deviation(b) for b in beta_values]


###################################################
### Statistics
###################################################

def roc(data):
    tpv, fnv, tnv, fpv, _, _ = data
    tpr = [tp * 1.0 / (tp + fn) for tp, fn in zip(tpv, fnv)]
    # calculate the false positive rate
    fpr = [fp * 1.0 / (fp + tn) for fp, tn in zip(fpv, tnv)]
    print('fpr, ', fpr)
    print('tpr, ', tpr)
    return fpr, tpr


def get_quantitative(A, B, W, show=True):
    """**A** is the referece, and **B** is the detected result, **W** is the
    whole set calculate the true positive, false negative, true negative and
    false positive
    """
    A = set(A)
    B = set(B)
    W = set(W)
    # no of true positive, no of elements belongs to B and also belongs to A
    tp = len(set.intersection(A, B))

    # no of false negative no of elements belongs to A but doesn't belong to B
    fn = len(A - B)

    # no of true negative, no of element not belongs to A and not belong to B
    tn = len(W - set.union(A, B))
    # no of false positive. no of ele. not belongs to A but belongs to B
    fp = len(B - A)

    # sensitivity is the probability of a alarm given that the this flow is
    # anormalous
    # sensitivity = tp * 1.0 / (tp + fn)
    # specificity is the probability of there isn't alarm given that the flow
    # is normal
    # specificity = tn * 1.0 / (tn + fp)

    tpr = tp * 1.0 / (tp + fn)
    fpr = fp * 1.0 / (fp + tn)

    # ret = tp, fn, tn, fp, sensitivity, specificity
    ret = tp, fn, tn, fp, fpr, tpr
    if show:
        OUT_STRING = """tp: %f\t fn: %f\t tn: %f\t fp: %f
        fpr: %f\ttpr: %f
        """
        print(OUT_STRING % ret)
    return ret






if __name__ == "__main__":
    # cal_corrcoef('./Result/merged_flows.csv', './Result/merged_npcor.pkz')
    # visualize_caida('./Result/merged_flows.csv')
    # cal_TDG('./Result/merged_flows.csv', './Result/merged_stored_TDG.pkz')
    # cal_TDG('./Result/merged_flows.csv', './Result/merged_stored_TDG.pk')
    ana_deg_distribution()
