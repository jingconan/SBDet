#!/usr/bin/env python
from __future__ import print_function, division
from SBDet import *
import pylab as P
import numpy as np
import networkx as nx
from subprocess import check_call
from Util import load, zload, dump

""" Analyze
"""


# Dependencies
# sigs_nx.pk // sigs stored in networkx format
# GCM_Tr.pk // solution --> this is unecessary, we will use equal weights
# flows.txt // the normal IP Address. We can stored it separately.

# this is used to get All ddos ip addresses. This tool should only deal with
# SIGs. botnet data should be converted to SIGs first.

def get_ips(data, format_=None):
    nnx = NetworkXGraph(data=data)
    if format_ is not None:
        return [format_(ip) for ip in nnx.get_vertices()]
    return nnx.get_vertices()

# GetAdjacent Matrix
ab_ids = range(200, 230)
sigs = load('./Result/sigs_nx.pk')
adj_mats = [nx.to_scipy_sparse_matrix(sigs[i]) for i in ab_ids]

#### Identify the Pivot Nodes ######
tr = load('./Result/GCM_tr.pk')
weights = tr['solution']
p_nodes = ident_pivot_nodes(adj_mats, weights, 0.8)

# select *SN* Nodes together with p_nodes
N = adj_mats[0].shape[0]
perm = np.random.permutation(N - len(p_nodes))
SN = 100
selected_nodes = perm[:(SN-len(p_nodes))]
selected_nodes = np.concatenate([selected_nodes, p_nodes])

# get Adjacent matrix for the selected subgraph
get_subarray = lambda x, y: x[y, :][:, y]
small_adj_mats = [get_subarray(adj, selected_nodes) for adj in adj_mats]

DDOS_DATA = './Result/flows.txt'
ddos_ips = get_ips(HDF_FlowExporter(DDOS_DATA), format_=np_to_dotted)
nodes = sigs[0].nodes()

selected_ips = [nodes[i] for i in selected_nodes]
selected_ddos_ips = set(selected_ips) & set(ddos_ips)

data = dict(ips=selected_ips,
            ddos_ips=selected_ddos_ips,
            adj_mats=small_adj_mats)

dump(data, './Result/small_prob.pk')

####

