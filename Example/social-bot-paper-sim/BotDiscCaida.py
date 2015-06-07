"""  Botnet Discovery for Caida dataset

Dependency
---------------
loc6-20070501-2055-40.sigs
ddostrace.20070804_134936-10.sigs
det_caida_divs.pkz

"""
from __future__ import print_function, division, absolute_import
import sys
sys.path.insert(0, "../../")
# from SBDet import parseToCoo, gen_sigs, mix, select_model
from SBDet import *
import pylab as P
from subprocess import check_call
import networkx as nx

bg_sigs, bg_nodes = parseToCoo('loc6-20070501-2055-40.sigs',
        undirected=True)
# bg_sigs = bg_sigs[0:300]
bg_sigs = bg_sigs[0:360]

deg_samples = mg_sample(len(bg_nodes), bg_sigs, 100, 50)
ER_para, ER_lk = mle(deg_samples, 'ER')




def symmetrize(a):
        # return a + a.T - P.diag(a.diagonal())
        return a + a.T

def convert(sigs):
    return [symmetrize(g.tocsr()) for g in sigs]



botnet_sigs, botnet_nodes = parseToCoo('ddostrace.20070804_134936-10.sigs',
                                       undirected=True)
# mix_sigs, mix_nodes = mix((bg_sigs, bg_nodes),
#         (botnet_sigs, botnet_nodes), 200)


# bg_sigs = [sig.tolil() for sig in bg_sigs]
# botnet_sigs = [sig.tolil() for sig in botnet_sigs]

mix_sigs, mix_nodes = mix_append((bg_sigs, bg_nodes),
        (botnet_sigs, botnet_nodes), 200)
# botnet_nodes = bg_nodes[:len(botnet_nodes)]

divs = monitor_deg_dis(mix_sigs, 'ER', (ER_para, 1e-10), minlength=None)


# div_data = zload('./det_caida_divs.pkz')
# div_data = zload('./det_caida_divs-360.pkz')
# divs = div_data['divs']
# P.plot(div_data['divs'])
# P.show()
# sys.exit(0)

# THRE = 0.19
THRE = 0.15

det_idx = [i for i, div in enumerate(divs) if div > THRE]

# def convert (g):
#     ms = g.todense()
#     return ms + ms.T
bot_adjs = [mix_sigs[idx] for idx in det_idx]


# convert bot_adjs to symmetric and to csr format
bot_adjs = convert(bot_adjs)

# def ident_pivot_nodes(bot_adjs, weights, thres):
# pivot_th = 0.2
# pivot_th = 0.4
pivot_th = 0.11
# pivot_th = 0.15
cor_th = 0.3
# w1 = 0.02
w1 = 2
w2 = 0.01
# w2 = 1
# w2 = 0
# w1 = 2
# w2 = 0.013
# w2 = 0.0012
# w2 = 0.013
# lamb = -100
lamb = 0

# botnet = detect_botnet(bot_adjs, pivot_th, cor_th, w1, w2, lamb)
node_num = bot_adjs[0].shape[0]
weights = np.ones((node_num, )) / node_num  # equal weights
sigs = bot_adjs
p_nodes, total_inta_mat = ident_pivot_nodes(bot_adjs, weights, pivot_th)
print('p_nodes', p_nodes)

# zdump(dict(p_nodes=p_nodes, total_inta_mat=total_inta_mat), './p_node_det.pkz')
# zdump(dict(p_nodes=p_nodes, total_inta_mat=total_inta_mat), './p_node_det-360.pkz')

# sorted_inta = P.array(sorted(total_inta_mat, reverse=True))
# sorted_inta = sorted_inta[sorted_inta>0]
# P.semilogy(sorted_inta)
# P.plot(sorted_inta)
# P.xlim([-1, len(sorted_inta)])
# P.show()

inta = cal_inta_pnodes(sigs, weights, p_nodes, total_inta_mat[p_nodes])
# print('inta', inta)
# sigs = [sig.tocsr() for sig in sigs]
A, npcor = cal_cor_graph(sigs, p_nodes, cor_th)
np.fill_diagonal(A, 0)
Asum = A.sum(axis=0)
none_iso_nodes, = Asum.nonzero()
# draw_graph(A,
#         igore_iso_nodes=True,
#         pic_show=False,
#         pos='graphviz',
#         with_labels=False,
#         node_size=100,
#         node_color='blue',
#         edge_color='grey',
#         alpha=0.9)
# import ipdb;ipdb.set_trace()

# print('na', na)
# import ipdb;ipdb.set_trace()
# shrink_map = dict(zip(range(node_num), range(node_num)))
shrink_map = dict(zip(range(len(none_iso_nodes)), none_iso_nodes))
A = A[np.ix_(none_iso_nodes, none_iso_nodes)]
# print(A.shape)
inta = inta[none_iso_nodes]
node_num = A.shape[0]

# zdump(dict(A=A, npcor=npcor), './cor-graph-360.pkz')

# import sys; sys.exit(0)

print('--> start to generate csdp problem')
P0, q0, W = com_det_reg(A, inta, w1, w2, lamb, out='./prob.sdpb')

print('--> start to solve csdp problem')
check_call(['csdp', './prob.sdpb', './botnet.sol'])

print('--> start to parse csdp solution')
Z, X = parse_CSDP_sol('botnet.sol', node_num + 1)
solution = randomization(X, P0, q0, sn=10000)

botnet_nodes_set = set(botnet_nodes)
ref_sol = []
for nv in none_iso_nodes:
    ip = mix_nodes[nv]
    if ip in botnet_nodes_set:
        ref_sol.append(1)
    else:
        ref_sol.append(0)

def eval_obj(fea_sol):
    fea_sol = np.asarray(fea_sol)
    return np.dot(np.dot(fea_sol.T, P0), fea_sol) + np.dot(q0, fea_sol)

import igraph
ig = igraph.Graph(directed=False)
ig.add_vertices(range(len(shrink_map)))
I, J = A.nonzero()
edges = zip(I, J)
ig.add_edges(edges)
res = ig.community_leading_eigenvector(clusters=2)
solution3 = res.membership

# res = ig.community_spinglass()

e1 = eval_obj(solution)
e3 = eval_obj(solution3)
e_ref = eval_obj(ref_sol)
print('SDP Solution: %f\nleading eigen value: %f\nreference: %f\n'
        % (e1, e3, e_ref))
# import ipdb;ipdb.set_trace()

inta_diff = np.dot(inta, solution)
print('inta_diff', inta_diff)
# solution = np.array(ref_sol)


botnet, = np.nonzero(solution > 0)
print('[%i] ips out of [%i] ips are detected as bots' %
      (len(botnet), node_num))


# whole_set = set(mix_nodes)
# whole_set = set(botnet_nodes) | set(bg_nodes)
# whole_set = set([mix_nodes[i] for i in none_iso_nodes])
# botnet_nodes = set(botnet_nodes) & whole_set
# det_ips = [mix_nodes[i] for i in botnet]
# det_ips = set([mix_nodes[shrink_map[i]] for i in botnet])
# det_ips = set([mix_nodes[i] for i in botnet])
# fp = det_ips - botnet_nodes
# print('[%i] ip address are wrongly detected as bots' % (len(fp)))
# md = (whole_set - det_ips) & botnet_nodes
# print('[%i] bots are missed' % (len(md)))
# import ipdb;ipdb.set_trace()

tr = dict(A=A, solution=solution, mix_nodes=mix_nodes,
        botnet_nodes=botnet_nodes, shrink_map=shrink_map,
        ref_sol=ref_sol, inta=inta)
        # botnet_nodes=botnet_nodes)
zdump(tr, 'viz-com-test.pkz')
