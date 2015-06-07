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

bg_sigs, bg_nodes = parseToCoo('loc6-20070501-2055-40.sigs',
        undirected=True)
# bg_sigs = bg_sigs[0:300]
bg_sigs = bg_sigs[0:360]

botnet_sigs, botnet_nodes = parseToCoo('ddostrace.20070804_134936-10.sigs',
                                       undirected=True)
mix_sigs, mix_nodes = mix((bg_sigs, bg_nodes),
        (botnet_sigs, botnet_nodes), 200)

# div_data = zload('./det_caida_divs.pkz')
div_data = zload('./det_caida_divs-360.pkz')
divs = div_data['divs']
# P.plot(div_data['divs'])
# P.show()
# sys.exit(0)

THRE = 0.19

det_idx = [i for i, div in enumerate(divs) if div > THRE]

# def convert (g):
#     ms = g.todense()
#     return ms + ms.T
bot_adjs = [mix_sigs[idx] for idx in det_idx]

# def ident_pivot_nodes(bot_adjs, weights, thres):
pivot_th = 0.15
cor_th = 0.2
w1 = 1
w2 = 1
lamb = 1

# botnet = detect_botnet(bot_adjs, pivot_th, cor_th, w1, w2, lamb)
node_num = bot_adjs[0].shape[0]
weights = np.ones((node_num, )) / node_num  # equal weights
sigs = bot_adjs
p_nodes, total_inta_mat = ident_pivot_nodes(bot_adjs, weights, pivot_th)
print('p_nodes', p_nodes)
# import ipdb;ipdb.set_trace()

inta = cal_inta_pnodes(sigs, weights, p_nodes)

zdump(dict(p_nodes=p_nodes, inta=inta), './inta-to-pivot-node-360.pkz')
print('inta', inta)
P.plot(inta)
P.show()
# A, npcor = cal_cor_graph(sigs, p_nodes, cor_th)
# P0, q0, W = com_det_reg(A, inta, w1, w2, lamb, out='./prob.sdpb')
# check_call(['csdp', './prob.sdpb', './botnet.sol'])

# Z, X = parse_CSDP_sol('botnet.sol', node_num + 1)
# solution = randomization(X, P0, q0)
# inta_diff = np.dot(inta, solution)
# print('inta_diff', inta_diff)

# botnet, = np.nonzero(solution > 0)
# print('[%i] ips out of [%i] ips are detected as bots' %
#       (len(botnet), node_num))


# det_ips = [mix_nodes[i] for i in botnet]
# print(set(det_ips) - set(botnet_nodes))






# P.plot(divs)
# P.show()



