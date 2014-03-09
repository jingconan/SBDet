#!/usr/bin/env python
from __future__ import print_function, division
from SBDet import *
import pylab as P
import networkx as nx
from subprocess import check_call
from SBDet.Util import load, zload

def get_ips(data, format_=None):
    nnx = NetworkXGraph(data=data)
    if format_ is not None:
        return [format_(ip) for ip in nnx.get_vertices()]
    return nnx.get_vertices()

ab_ids = range(200, 230)
sigs = load('./Result/sigs_nx.pk')
adj_mats = [nx.to_scipy_sparse_matrix(sigs[i]) for i in ab_ids]

#### Identify the Pivot Nodes ######
tr = load('./Result/GCM_tr.pk')
weights = tr['solution']
p_nodes = ident_pivot_nodes(adj_mats, weights, 0.8)

#### Calculate interactions of nodes with pivot nodes ####
inta = cal_inta_pnodes(adj_mats, tr['solution'], p_nodes)

#### Calculate the correlation graph ####
A, npcor = cal_cor_graph(adj_mats, p_nodes, 0.2)

w2 = 0.01
P0, q0, W = com_det_reg(A, inta, w1=0, w2=w2, lamb=0, out='./prob.sdpb')
# check_call('./csdp6.1.0linuxp4/bin/csdp ./prob.sdpb ./botnet.sol',
#            shell=True)

node_num = len(inta)
solution = randomization(X, P0, q0)
solution = solution[0:-1]
inta_diff = np.dot(inta, solution)
print('inta_diff', inta_diff)

botnet, = np.nonzero(solution > 0)
print('[%i] ips out of [%i] ips are detected as bots' %
      (len(botnet), node_num))
nodes = sigs[ab_ids[0]].nodes()
botnet_ips = [nodes[b] for b in botnet]
bot_ips_file = './Result/reg_detected_ips_w2_%s.pkz' % (w2)
zdump(botnet_ips, bot_ips_file)


###### Statistics
detected_ips = zload(bot_ips_file)
# NORMAL_DATA = './Result/sampled_dump.pkz'
NORMAL_DATA = './Result/sampled_data.pkz'
DDOS_DATA = './Result/flows.txt'


normal_ips = get_ips(HDF_DumpFS(NORMAL_DATA), format_=np_to_dotted)
ddos_ips = get_ips(HDF_FlowExporter(DDOS_DATA), format_=np_to_dotted)

data = get_quantitative(ddos_ips, detected_ips,
                        set(normal_ips) | set(ddos_ips), show=True)



#####
import ipdb;ipdb.set_trace()
