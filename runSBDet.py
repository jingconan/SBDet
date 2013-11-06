#! /usr/bin/env python
#! /home/wangjing/Apps/python/bin/python
####
from __future__ import print_function, division
from SBDet import *
import pylab as P
import networkx as nx
from subprocess import check_call


#### Calculate Social Interaction Graph ####
# sigs = cal_SIG('./Result/merged_flows.csv', tp='igraph')
# animate_SIGs(sigs, './Animation/')
# dump(sigs, './Result/sigs.pkz')
# sigs = cal_SIG('./Result/merged_flows.csv', tp='networkx')
# dump(sigs, './Result/sigs_nx.pk')


#### Verify whether normal SIGs satisfies ERGM ####
# sigs = load('./Result/sigs.pkz')
# bv = P.linspace(0.0005, 1, 100)
# deviations = verify_ERGM(sigs[0:200], 'igraph', beta_values=bv)
# P.plot(bv, deviations)
# P.grid(which='major')
# P.title('K-L divergence of normal degree dist and ERGM')
# P.xlabel('$\\beta$')
# P.ylabel('K-L divergence')
# P.savefig('./Result/fitness_normal_dd_ERGM.pdf')
# P.show()

#### Monitor the degree distribution ####
# sigs = load('./Result/sigs.pkz')
# divs_KL = monitor_deg_dis(sigs, sigs[0:200], 'igraph', KL_div)
# divs_LDP = monitor_deg_dis(sigs, sigs[0:200], 'igraph', LDP_div)
# P.subplot(211)
# P.plot(divs_KL)
# P.title('K-L div. of all windows')
# P.ylabel('K-L divergence')
# P.subplot(212)
# P.plot(divs_LDP)
# P.title('LDP div. of all windows')
# P.ylabel('LDP divergence')
# P.xlabel('time (s)')
# P.savefig('./Result/comparison_KL_LDP.pdf')
# P.show()


#### Estimate the Generalized Configuration Model (GCM) ####
sigs = load('./Result/sigs_nx.pk')
ab_sigs = sigs[200:230]
adj_mats = [nx.to_scipy_sparse_matrix(sig) for sig in ab_sigs]

# tr = EstTrafProb(adj_mats)
# dump(tr, './Result/GCM_tr.pk')

#### Identify the Pivot Nodes ######
tr = load('./Result/GCM_tr.pk')
weights = tr['solution']
p_nodes = ident_pivot_nodes(adj_mats, weights, 0.8)

# print('p_nodes', p_nodes)


#### Calculate interactions of nodes with pivot nodes ####
inta = cal_inta_pnodes(adj_mats, tr['solution'], p_nodes)

# P.plot(inta)
# P.xlabel('node seq.')
# P.ylabel('interaction with pivot nodes')
# P.savefig('./Result/inta_with_pnodes.pdf')
# P.show()


#### Calculate the correlation graph ####
# A, npcor = cal_cor_graph(adj_mats, p_nodes, 0.2)
# nxg = nx.from_numpy_matrix(A)
# save_graph(nxg, './Result/cor_graph_0.2.pdf')

#### Community Detection #######
# with open('./prob.sdpb', 'w') as f:
#     com_det(A, inta, w=10, lamb=10, out=f)


# check_call['./csdp6.1.0linuxp4/bin/csdp ./prob.sdpb ./botnet.sol',
#            shell=True]


##### Randomization to Generate Feasible Solution #######
# Z, X = parse_CSDP_sol('./csdp6.1.0linuxp4/bin/botnet.sol', 1170)
Z, X = parse_CSDP_sol('./botnet.sol', 1170)
solution = randomization(X)
inta_diff = np.dot(inta, solution)
print('inta_diff', inta_diff)
botnet, = np.nonzero(solution == P.sign(inta_diff))
print('botnet', botnet)
nodes = ab_sigs[0].nodes()
botnet_ips = [nodes[b] for b in botnet]

dump(botnet_ips, './Result/botnet_ips.pk')


#### Count Accuracy of the Result ######
botnet_ips = load('./Result/botnet_ips.pk')
NORMAL_DATA = './Result/sampled_dump.pkz'
DDOS_DATA = './Result/flows.txt'
def get_ips(data):
    nnx = NetworkXGraph(data=data)
    return nnx.get_vertices()

# normal_ips = get_ips(HDF_DumpFS(NORMAL_DATA))
ddos_ips = get_ips(HDF_FlowExporter(DDOS_DATA))
ddos_ips = [np_to_dotted(ip) for ip in ddos_ips]

ddos_set = set(ddos_ips)
detect_set = set(botnet_ips)
mis_ratio = len(ddos_set - detect_set) * 1.0 / len(ddos_set)
correct_ratio = len(ddos_set & detect_set) * 1.0 / len(ddos_set)
print('correct_ratio', correct_ratio)
print('mis_ratio', mis_ratio)

# P.plot(solution, '*')
# P.show()
########################
