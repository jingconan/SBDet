#!/usr/bin/env python
from __future__ import print_function, division, absolute_import
import os
import sys
import scipy as sp
import numpy as np
# import numpy as np
# import networkx as nx
# import matplotlib.pyplot as plt
# from collections import Counter

from .Data import HDF_FlowExporter
# from Dataset import HDF_FlowExporter
from .CGraph import Igraph
from .CGraph import NetworkXGraph
from .Util import dump, load
# from .Util import I1
# from .Util import np_index, np_to_dotted
from .Util import progress_bar
from .Util import DataEndException
from .Util import igraph
# from .Util import sp, la, stats


""" Calculate Social Interaction Graph (SIGs) from **data file**, which
    should be the subclass of the **Data** class Look at Data.py for more
    info about the supported file types
"""


def cal_SIG_low_mem(data_file, interval=10.0, dur=10.0,
                    rg=(0.0, float('inf')), folder=None):
    """ Calculate the Social Interaction Graph (SIG)

    Parameters
    ---------------
    data_file : Data class
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
            sys.stdout.write("\r[%d]" % i)
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


def to_sigs(data, out_folder, dur):
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    cal_SIG_low_mem(data,
                    interval=dur,
                    dur=dur,
                    folder=out_folder)
    pack_sigs(out_folder, out_folder+'sigs.pk')


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
                                rg=[start_time + i * interval,
                                    start_time + i * interval + dur],
                                rg_type='time')
            edges = tg.get_edges(records)
            tg.add_edges(edges)
            sigs.append(tg.graph)
    except DataEndException:
        print('reach end')

    return sigs


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


""" Load SIGs calculated by pcap2sigs tool. Please go to
https://github.com/hbhzwj/pcap2sigs for more info.
"""


def parseToLil(f_name, first_k=None):
    with open(f_name, 'r') as fid:
        line = fid.readline()
        nodes = line.split()
        if first_k:
            nodes = nodes[:first_k]
        sigs = []
        g = []
        line = fid.readline()
        for line in fid:
            if line[0] == 'G':
                sigs.append(g)
                g = []
                continue

            res = line.split(' -> ')
            from_node = int(res[0])
            to_node = int(res[1])
            if first_k and (from_node >= first_k or to_node >= first_k):
                continue
            g.append((from_node, to_node))
        sigs.append(g)
    return sigs, nodes

def parseToCoo(f_name, undirected=False, first_k=None):
    """  parse the output of pcap2sigs to scipy.sparse.coo_matrix

    Parameters
    ---------------
    f_name : str
        file name of the pcap2sigs output
    undirected : bool
        if undirected is true, then the graph is undirecte, the return value
            will be up-trilangular matrix.
    first_k : int
        if first_k is set, each sigs is subgraph of the first k elements

    Returns
    --------------
    sparse_sigs : list of scipy.sparse.coo_matrix
        if undirected is true, then each coo_matrix is up-trilangular
    nodes : list of str
        the ip address of all nodes
    """
    sigs, nodes = parseToLil(f_name, first_k)
    sparse_sigs = []
    g_size = len(nodes)
    for g in sigs:
        N = len(g)
        if N == 0:
            mat = sp.sparse.coo_matrix((g_size, g_size), dtype=np.int)
        else:
            I, J = zip(*g)
            if undirected:
                IJMat = np.array([I, J], copy=True)
                I = np.min(IJMat, axis=0)
                J = np.max(IJMat, axis=0)
            mat = sp.sparse.coo_matrix((sp.ones(N,), (I, J)),
                    shape=(g_size, g_size))
        sparse_sigs.append(mat)
    return sparse_sigs, nodes


"""  Generate Validiation Dataset of SIGs
"""

import copy
def union_nodes(nodes1, nodes2):
    """  Union two nodes.

    Parameters
    ---------------
    Returns
    --------------
    nodes : list
        union set of nodes1 and nodes2. The first len(nodes1) nodes in *nodes*
        comes from nodes1.

    map2 : map from int to int
        map2 is the map from idx in nodes2 to idx in **nodes**
    """
    nodes = copy.copy(nodes1)
    N1 = len(nodes1)
    nodes1_map = dict(zip(nodes1, range(N1)))
    map2 = dict()
    i = N1
    for j, n in enumerate(nodes2):
        idx = nodes1_map.get(n)
        if idx is None: # if it doesn't exist in nodes1
            nodes.append(n)
            map2[j] = i
            i += 1
            pass
        else:
            map2[j] = idx # share the same node

    return nodes, map2

def mix(dataset1, dataset2, start):
    """ Mix two sigs dataset.

    Parameters
    ---------------
    dataset1, dataset2 : tuple
        dataset1 = (sigs1, nodes1) and dataset2 = (sigs2, nodes2) are two datasets
        of SIGs.  It is assumed that len(sigs2) < len(sigs1). This function will
        mix sigs2 with a segment of sigs1 and return the mixed dataset

        sigs1 and sigs2 are assumed to be a list of scipy.sparse matrix

    start : int
        the start position of mix. dataset2, will be mixed with
        sigs1[start:(start+len(sigs2))]
    Returns
    --------------
    sigs : a list of scipy.sparse.coo_matrix
        mixed sigs dataset
    nodes : list
        union of nodes1 and nodes2
    """
    s1, n1 = dataset1
    s2, n2 = dataset2
    assert(len(s2) <= len(s1))

    nodes, map2 = union_nodes(n1, n2)
    g_size = len(nodes)

    def map_sig(sig, map_):
        I1, J1 = sig.nonzero()
        I = [map_[x] for x in I1]
        J = [map_[x] for x in J1]
        return I, J

    sigs = copy.deepcopy(s1[:start])
    for k in xrange(start, start + len(s2)):
        Im1, Jm1 = s1[k].nonzero()
        Im2, Jm2 = map_sig(s2[k-start], map2)
        M = len(Im1) + len(Im2)
        mat = sp.sparse.coo_matrix((sp.ones(M,), \
                (Im1.tolist() + Im2, Jm1.tolist() + Jm2)),
                shape=(g_size, g_size))
        sigs.append(mat)
    sigs.extend(s1[(start + len(s2)):])
    return sigs, nodes
